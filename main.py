import asyncio
import enum
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, Union, Callable, Any, Awaitable, get_args, Tuple, Optional, ForwardRef, get_origin

from typing import TypeVar, Annotated, Literal

T = TypeVar('T')


@dataclass
class Provide:
    key: str = None


class DependencyKey:
    Service: str = 'service'
    Request: str = 'request'
    Response: str = 'response'
    Session: str = 'session'
    Query: str = 'query'
    Body: str = 'body'
    Params: str = 'params'

    @classmethod
    def to_list(cls) -> list:
        return [cls.Service, cls.Request, cls.Response, cls.Session, cls.Query, cls.Body, cls.Params]


class DependencyMeta:

    def __init__(self, metadata=None):
        self.metadata = metadata

    def __getitem__(self, params: Union[T, Tuple[Union[T, Any]]]) -> Annotated[T, Any]:
        if isinstance(params, tuple):
            if len(params) == 2:
                type_annotation, metadata = params
                return Annotated[type_annotation, self.metadata]
            elif len(params) == 1:
                return Annotated[params[0], self.metadata]
            else:
                raise ValueError("Inject[type, metadata] syntax is required")
        else:
            if isinstance(params, str):
                gl = globals()
            return Annotated[params, self.metadata]


Inject = DependencyMeta(DependencyKey.Service)
Req = DependencyMeta(DependencyKey.Request)
Res = DependencyMeta(DependencyKey.Response)
Session = DependencyMeta(DependencyKey.Session)
Query = DependencyMeta(DependencyKey.Query)
Body = DependencyMeta(DependencyKey.Body)
Params = DependencyMeta(DependencyKey.Params)


class Scope(enum.Enum):
    Request = 'Request'
    Transient = 'Transient'
    Singleton = 'Singleton'


class ModuleMetadata:
    Imports: str = 'imports'
    Exports: str = 'exports'
    Providers: str = 'providers'
    Controllers: str = 'controllers'
    Global: str = '__global__'
    Root: str = '__root__'


class ClassMetadata:
    Metadata = '__dependency_metadata__'
    _global_providers = []
    _module = None

    def __init__(self, module: Callable, global_providers: list = None):
        self._module = module
        self._global_providers = global_providers or []

    def get_module(self):
        return self._module

    def get_service_providers(self):
        providers = self._global_providers + getattr(self._module, ModuleMetadata.Providers, [])
        import_providers = []
        # Only not a root module need to get import_providers to share
        if not getattr(self._module, ModuleMetadata.Root, False):
            for im in getattr(self._module, ModuleMetadata.Imports, []):
                import_providers = import_providers + getattr(im, ModuleMetadata.Exports, [])
        return providers, import_providers


class Proxy:
    def __init__(self, service: Type):
        self.service = service

    def __getitem__(self, item: Any):
        return getattr(self.service, item)


class InjectorKey(enum.Enum):
    request: str = '_request'
    response: str = '_response'
    query_params: str = '_query_params'
    params: str = '_params'
    session: str = '_session'
    bodY: str = '_body'


def _uniq(data: list) -> list:
    return list(set(data))


class Injector:
    _instance: "Injector" = None
    _services = {}
    _global_service_instances = {}
    _singleton_instances = {}
    _singleton_classes = set()
    _request = None
    _response = None
    _query_params = {}
    _params = {}
    _session = {}
    _body = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Injector, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def get_instance(cls, *args, **kwargs):
        return Injector(*args, **kwargs)

    def add_transient(self, service: Type):
        self._services[service] = service

    def add_singleton(self, service: Type):
        self._services[service] = service
        self._singleton_classes.add(service)

    def add_singleton_instance(self, service: Union[Type, str], service_instance: object):
        self._singleton_instances[service] = service_instance

    def set_value(self, key: InjectorKey, value: object):
        setattr(self, key.value, value)

    @classmethod
    def get_dependency_metadata(cls, service: Union[Type, object]) -> list:
        metadata: ClassMetadata = getattr(service, ClassMetadata.Metadata, None)
        if metadata is not None:
            providers, import_providers = metadata.get_service_providers()
            return [m.provide if isinstance(m, ModuleProviderDict)
                    else m for m in _uniq(providers + import_providers)]
        raise ValueError(f"Dependency Metadata not found  for {service.__name__} service ")

    @classmethod
    def _get_type_from_annotation(cls, annotation: Any):
        args: tuple = get_args(annotation)
        # check if key is from provide(ModuleProviderDict)
        if len(args) == 2:
            if isinstance(args[0], Provide):
                return args[0].key, args[1]
            return args[0], args[1]
        else:
            if isinstance(annotation, Provide):
                return annotation.key, None
            return annotation, None

    def _resolve_outer_service(self, annotation: Any, dep_key: DependencyKey):
        match dep_key:
            case DependencyKey.Request:
                return self._request
            case DependencyKey.Response:
                return self._response
            case DependencyKey.Body:
                return self._body
            case DependencyKey.Session:
                return self._session[annotation] or None
            case DependencyKey.Params:
                return self._params[annotation] or None
            case DependencyKey.Query:
                return self._query_params[annotation] or None
            case _:
                return None

    async def _resolve_module_provider_dict(self, instance: "ModuleProviderDict", search_scope: list):
        if instance.value:
            return instance.value
        elif instance.factory:
            return await self.resolve_factory(
                factory=instance.factory,
                inject=instance.inject,
                search_scope=search_scope
            )
        elif instance.existing:
            return await self.get(instance.existing)
        else:
            return None

    async def _check_exist_singleton(self, key: Union[Type, str]):
        if key in self._singleton_instances:
            instance = self._singleton_instances[key]
            # to keep improve
            if isinstance(instance, ModuleProviderDict):
                search_scope = self.get_dependency_metadata(instance)
                if instance.provide in search_scope:
                    return await self._resolve_module_provider_dict(instance, search_scope=search_scope)
                else:
                    raise ValueError(
                        f"Service {instance.__class__.__name__} "
                        f"not found in scope")
            else:
                return instance
        return None

    def _check_service(self, key: Union[Type, str], origin: Optional[list] = None) -> tuple:
        if key not in self._services:
            raise ValueError(f"Service {key} not found")
        service = self._services[key]
        if service in (origin or []):
            raise ValueError(f"Circular dependency found  for {service.__name__} service ")
        return service, origin or set()

    async def _resolve_property(self, key: Union[Type, str], origin: Optional[list] = None):
        service, origin = self._check_service(key, origin)
        search_scope = self.get_dependency_metadata(service)
        origin.add(service)
        annotations: dict = getattr(service, '__annotations__', {})
        for name, param_annotation in annotations.items():
            annotation, dep_key = self._get_type_from_annotation(param_annotation)
            if isinstance(annotation, ForwardRef):
                annotation = eval(annotation.__forward_arg__, globals())
                if annotation is None:
                    raise ValueError(f"Unknown forward reference: {annotation}")
            if dep_key in DependencyKey.to_list():
                if dep_key is not DependencyKey.Service:
                    dependency = self._resolve_outer_service(annotation, dep_key)
                    setattr(service, name, dependency)
                elif annotation in search_scope:
                    dependency = await self.get(annotation, origin=origin)
                    setattr(service, name, dependency)
                else:
                    _name: str = annotation.__name__ if not isinstance(annotation, str) else annotation
                    raise ValueError(
                        f"Service {_name} "
                        f"not found in scope of {service.__name__}")
            else:
                continue
        origin.remove(service)
        self._services[key] = service

    async def _get_method_dependency(self, method_to_resolve: Callable, search_scope: list, origin: list):
        params = inspect.signature(method_to_resolve).parameters
        args = {}
        for name, param in params.items():
            if name != 'self' and param.annotation is not inspect.Parameter.empty:
                annotation, dep_key = self._get_type_from_annotation(param.annotation)
                if dep_key is not DependencyKey.Service:
                    dependency = self._resolve_outer_service(annotation, dep_key)
                    args[name] = dependency
                elif annotation in search_scope:
                    dependency = await self.get(annotation, origin=origin)
                    args[name] = dependency
                else:
                    _name: str = annotation.__name__ if not isinstance(annotation, str) else annotation
                    raise ValueError(f"Service {_name} not found in scope {search_scope}")
        return args

    @classmethod
    async def _call_method(cls, method: Callable, args: dict):
        if inspect.iscoroutinefunction(method):
            return await method(**args)
        return method(**args)

    async def resolve_factory(self, factory: Callable, inject: list, search_scope: list):
        search_scope_by_inject = [m for m in inject if m in search_scope]
        args = await self._get_method_dependency(method_to_resolve=factory, search_scope=search_scope_by_inject,
                                                 origin=[])
        return await self._call_method(method=factory, args=args)

    async def _resolve_method(self, key: Union[Type, str, object], method: str = '__init__',
                              origin: Optional[list] = None):
        service, origin = self._check_service(key, origin)
        search_scope = self.get_dependency_metadata(service)
        origin.add(service)
        method_to_resolve = getattr(service, method, None)
        if not method_to_resolve:
            raise Exception(f"Method {method}  not found in {service.__name__} service ")
        args = await self._get_method_dependency(method_to_resolve, search_scope, origin)
        if method == '__init__':
            result = service(**args)
        else:
            # Service must be an instance (controller)
            instance = await self.get(key)
            instance_method = getattr(instance, method, method_to_resolve)
            result = await self._call_method(instance_method, args)
        if service in self._singleton_classes and method == '__init__':
            self._singleton_instances[service] = result
        origin.remove(service)
        return result

    async def get(self, key: Union[Type, str], method: str = '__init__', origin: Optional[list] = None):
        exist_singleton = await self._check_exist_singleton(key=key)
        if exist_singleton:
            if method == '__init__':
                return exist_singleton
            value = await self._resolve_method(key, method=method, origin=origin)
        else:
            await self._resolve_property(key, origin=origin)
            value = await self._resolve_method(key, method=method, origin=origin)
        return value


HTTPMethod = Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']


class RouteKey:
    path: str = '__path__'
    kwargs: str = '__kwargs__'
    method: str = '__method__'


# Decorator
class Injectable:
    scope: Scope = None

    def __init__(self, scope: Scope = Scope.Singleton):
        self.scope = scope
        self.container = Injector.get_instance()

    def __call__(self, cls: Type):
        match self.scope:
            case Scope.Transient:
                self.container.add_transient(cls)
            case Scope.Request:
                self.container.add_transient(cls)
            case _:
                self.container.add_singleton(cls)
        return cls


class Controller:
    def __init__(self, path: str = '/', **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.container = Injector.get_instance()

    def __call__(self, cls, **kwargs):
        self.container.add_singleton(cls)
        # put path and kwargs in controller property
        setattr(cls, RouteKey.path, self.path)
        setattr(cls, RouteKey.kwargs, self.kwargs)
        return cls


class Route:
    def __init__(self, path: str = '', method: HTTPMethod = 'GET', **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.method = method

    def __call__(self, handler):
        # put path and kwargs in controller handler
        setattr(handler, RouteKey.path, self.path)
        setattr(handler, RouteKey.kwargs, self.kwargs)
        setattr(handler, RouteKey.method, self.method)
        return handler


class Module:
    providers: list[Callable] = []
    controllers: list[Callable] = []
    imports: list[Callable] = []
    exports: list[Callable] = []
    is_global: bool = False

    def __init__(
            self,
            providers: list[Callable] = None,
            controllers: list[Callable] = None,
            imports: list[Callable] = None,
            exports: list[Callable] = None,
            is_global: bool = False
    ):
        self.providers = providers or []
        self.controllers = controllers or []
        self.imports = imports or []
        self.exports = exports or []
        self.is_global = is_global

    def __call__(self, cls):
        setattr(cls, ModuleMetadata.Providers, self.providers)
        setattr(cls, ModuleMetadata.Controllers, self.controllers)
        setattr(cls, ModuleMetadata.Imports, self.imports)
        setattr(cls, ModuleMetadata.Exports, self.exports)
        setattr(cls, ModuleMetadata.Global, self.is_global)
        return cls


class ModuleProviderDict:
    inject: list = []
    provide: Union[str, Type]
    value: Any = None
    factory: Callable[..., Union[Awaitable, Any]] = None
    existing: Any = None

    def __init__(
            self,
            provide: Union[str, Type],
            value: Any = None,
            factory: Callable[..., Union[Awaitable, Any]] = None,
            existing: Any = None,
            inject: list = None
    ):
        self.provide = provide
        self.value = value
        self.factory = factory
        self.existing = existing
        self.inject = inject or []
        Injector.get_instance().add_singleton_instance(provide, self)


# Compiler
class Compiler(ABC):

    def __init__(self, module, global_data=None, is_root=False):
        self.module = module
        self.is_root = is_root
        self.global_data = global_data or []

    @abstractmethod
    def _extract(self) -> list:
        pass

    @abstractmethod
    def _type(self) -> Type["Compiler"]:
        pass

    def extract_providers(self) -> list:
        return _uniq(getattr(self.module, ModuleMetadata.Providers, []))

    def _put_dependency_metadata(self) -> None:
        data = self._extract()
        for p in data:
            global_data = self.extract_providers() if self.is_root else self.global_data
            # only p that have not metadata
            if not hasattr(p, ClassMetadata.Metadata):
                setattr(
                    p,
                    ClassMetadata.Metadata,
                    ClassMetadata(
                        self.module,
                        global_providers=global_data
                    )
                )

    def _extract_import(self) -> list:
        """
        Extract module imported in Module
        :return: list
        """
        return _uniq(getattr(self.module, ModuleMetadata.Imports, []))

    def compile(self) -> None:
        """
        Compile is about put module parent to the dependency metadata of a provider or controller
        :return:
        """
        if self.is_root:
            setattr(self.module, ModuleMetadata.Root, True)
            self.global_data = self.extract_providers()
            # compile global first
            imports = self._extract_import()
            not_global_module = []
            for im in imports:
                if not getattr(im, ModuleMetadata.Global, False):
                    not_global_module.append(im)
                else:
                    self._type()(im, global_data=self.global_data).compile()
                    self.global_data = self.global_data + getattr(im, ModuleMetadata.Exports, [])
            # compile non global after
            for im in not_global_module:
                self._type()(im, global_data=self.global_data).compile()
            self._put_dependency_metadata()
        else:
            self._put_dependency_metadata()


class ProviderCompiler(Compiler):
    def _extract(self) -> list:
        return _uniq(getattr(self.module, ModuleMetadata.Providers, []))

    def _type(self) -> Type["Compiler"]:
        return ProviderCompiler


class ControllerCompiler(Compiler):
    def _extract(self) -> list:
        return getattr(self.module, ModuleMetadata.Controllers, [])

    def _type(self) -> Type["Compiler"]:
        return ControllerCompiler


class ModuleCompiler(Compiler):
    def _extract(self) -> list:
        return getattr(self.module, ModuleMetadata.Providers, [])

    def _type(self) -> Type["Compiler"]:
        return ModuleCompiler

    @classmethod
    def _is_global(cls, module) -> bool:
        return getattr(module, ModuleMetadata.Global, False)

    @classmethod
    def extract_imports(cls, module) -> list:
        return _uniq(getattr(module, ModuleMetadata.Exports, []))

    def _extend_providers(self, extends: list = None):
        setattr(self.module, ModuleMetadata.Providers, _uniq((extends or []) + self._extract()))

    def compile(self) -> None:
        # extract provider form all global module
        # but before, provider is need to compiled to put metadata of module in providers
        imports = self._extract_import()
        # global_modules = [m for m in imports if self._is_global(m)]
        # update providers to get all providers from imported modules
        imports = self._extract_import()
        for im in imports:
            if self._is_global(im):
                # Extends providers of root nodule to be global
                exports = self.extract_imports(im)
                self._extend_providers(exports)


# ==> Compile Module => Compile Providers => Compile Controller


# END Compiler
# Usage

@Injectable()
class GlobalService:
    test: Inject[Provide('TEST')]
    pass


@Injectable()
class GlobalService2:
    g: Inject[GlobalService]
    pass


@Injectable()
class ExportedService:
    pass


@Injectable(scope=Scope.Transient)
class DatabaseService:
    def __init__(self):
        ...


@Injectable(scope=Scope.Transient)
class LoggerService:
    ex_g_service: Inject[ExportedService]

    def log(self, data):
        print(f"{self.__class__.__name__}::{data}")


@Injectable()
class UserService2:
    _logger_service: Inject[LoggerService]
    _g_service: Inject[GlobalService]
    service: Inject['UserService']

    def log2(self, data: str):
        self._logger_service.log(data)


@Injectable()
class UserService:
    _second: Inject[UserService2]
    _db_service: Inject[DatabaseService]
    request: Req[Any]
    response: Res[Any]
    _logger_service: Inject[LoggerService]

    def log(self, data: str):
        self._logger_service.log(data)
        self._second.log2(data)


@Controller()
class UserController:
    service: Inject[UserService]
    g_service: Inject[GlobalService]
    ex_g_service: Inject[ExportedService]
    test2: Inject[Provide('TEST2')]

    @Route(method='GET')
    def get(self, request: Req[Any], res: Res[Any], log: Inject[LoggerService], test: Inject[Provide('TEST')]) -> str:
        return self.test2


@Module(
    providers=[DatabaseService],
    exports=[DatabaseService]
)
class DatabaseModule:
    ...


@Module(
    providers=[LoggerService],
    exports=[LoggerService]
)
class LogModule:
    g2: Inject[GlobalService2]
    ...


@Module(
    providers=[UserService, UserService2],
    imports=[DatabaseModule, LogModule],
    controllers=[UserController]
)
class UserModule:
    ...


@Module(
    providers=[ExportedService],
    exports=[ExportedService],
    is_global=True
)
class ExportedGlobalModule:
    ...


def example_factory(dep: Inject[GlobalService]):
    return "TEST2"


@Module(
    providers=[
        ModuleProviderDict(
            provide='TEST',
            value='TEST'
        ),
        GlobalService,
        ModuleProviderDict(
            provide='TEST2',
            factory=example_factory,
            inject=[GlobalService]
        ),
        GlobalService2
    ],
    imports=[
        ExportedGlobalModule,
        DatabaseModule,
        LogModule,
        UserModule
    ]
)
class AppModule:
    ...


if __name__ == '__main__':
    ProviderCompiler(AppModule, is_root=True).compile()
    ControllerCompiler(AppModule, is_root=True).compile()
    container = Injector.get_instance()

    user_service: UserService = asyncio.get_event_loop().run_until_complete(container.get(UserService))
    user_service.log('Test')

    controller: UserController = asyncio.get_event_loop().run_until_complete(container.get(UserController))
    response: str = asyncio.get_event_loop().run_until_complete(container.get(UserController, method='get'))
    print(response)
