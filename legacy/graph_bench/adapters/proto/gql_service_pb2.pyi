from graph_bench.adapters.proto import gql_types_pb2 as _gql_types_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResetTarget(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESET_ALL: _ClassVar[ResetTarget]
    RESET_SCHEMA: _ClassVar[ResetTarget]
    RESET_GRAPH: _ClassVar[ResetTarget]
    RESET_TIME_ZONE: _ClassVar[ResetTarget]
    RESET_PARAMETERS: _ClassVar[ResetTarget]

class ResultType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BINDING_TABLE: _ClassVar[ResultType]
    GRAPH: _ClassVar[ResultType]
    OMITTED: _ClassVar[ResultType]

class TransactionMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    READ_WRITE: _ClassVar[TransactionMode]
    READ_ONLY: _ClassVar[TransactionMode]
RESET_ALL: ResetTarget
RESET_SCHEMA: ResetTarget
RESET_GRAPH: ResetTarget
RESET_TIME_ZONE: ResetTarget
RESET_PARAMETERS: ResetTarget
BINDING_TABLE: ResultType
GRAPH: ResultType
OMITTED: ResultType
READ_WRITE: TransactionMode
READ_ONLY: TransactionMode

class HandshakeRequest(_message.Message):
    __slots__ = ("protocol_version", "credentials", "client_info")
    class ClientInfoEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    CLIENT_INFO_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    credentials: _gql_types_pb2.AuthCredentials
    client_info: _containers.ScalarMap[str, str]
    def __init__(self, protocol_version: _Optional[int] = ..., credentials: _Optional[_Union[_gql_types_pb2.AuthCredentials, _Mapping]] = ..., client_info: _Optional[_Mapping[str, str]] = ...) -> None: ...

class HandshakeResponse(_message.Message):
    __slots__ = ("protocol_version", "session_id", "server_info", "limits")
    class LimitsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_INFO_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    session_id: str
    server_info: ServerInfo
    limits: _containers.ScalarMap[str, int]
    def __init__(self, protocol_version: _Optional[int] = ..., session_id: _Optional[str] = ..., server_info: _Optional[_Union[ServerInfo, _Mapping]] = ..., limits: _Optional[_Mapping[str, int]] = ...) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("name", "version", "features")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    features: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., features: _Optional[_Iterable[str]] = ...) -> None: ...

class ConfigureRequest(_message.Message):
    __slots__ = ("session_id", "schema", "graph", "time_zone_offset_minutes", "parameter")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    TIME_ZONE_OFFSET_MINUTES_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    schema: str
    graph: str
    time_zone_offset_minutes: int
    parameter: SessionParameter
    def __init__(self, session_id: _Optional[str] = ..., schema: _Optional[str] = ..., graph: _Optional[str] = ..., time_zone_offset_minutes: _Optional[int] = ..., parameter: _Optional[_Union[SessionParameter, _Mapping]] = ...) -> None: ...

class SessionParameter(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: _gql_types_pb2.Value
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[_gql_types_pb2.Value, _Mapping]] = ...) -> None: ...

class ConfigureResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("session_id", "target")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    target: ResetTarget
    def __init__(self, session_id: _Optional[str] = ..., target: _Optional[_Union[ResetTarget, str]] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CloseSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseSessionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PingRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class PongResponse(_message.Message):
    __slots__ = ("timestamp",)
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    timestamp: int
    def __init__(self, timestamp: _Optional[int] = ...) -> None: ...

class ExecuteRequest(_message.Message):
    __slots__ = ("session_id", "statement", "parameters", "transaction_id")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _gql_types_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_gql_types_pb2.Value, _Mapping]] = ...) -> None: ...
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    STATEMENT_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    statement: str
    parameters: _containers.MessageMap[str, _gql_types_pb2.Value]
    transaction_id: str
    def __init__(self, session_id: _Optional[str] = ..., statement: _Optional[str] = ..., parameters: _Optional[_Mapping[str, _gql_types_pb2.Value]] = ..., transaction_id: _Optional[str] = ...) -> None: ...

class ExecuteResponse(_message.Message):
    __slots__ = ("header", "row_batch", "summary")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    ROW_BATCH_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    header: ResultHeader
    row_batch: RowBatch
    summary: ResultSummary
    def __init__(self, header: _Optional[_Union[ResultHeader, _Mapping]] = ..., row_batch: _Optional[_Union[RowBatch, _Mapping]] = ..., summary: _Optional[_Union[ResultSummary, _Mapping]] = ...) -> None: ...

class ResultHeader(_message.Message):
    __slots__ = ("result_type", "columns", "ordered")
    RESULT_TYPE_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    ORDERED_FIELD_NUMBER: _ClassVar[int]
    result_type: ResultType
    columns: _containers.RepeatedCompositeFieldContainer[ColumnDescriptor]
    ordered: bool
    def __init__(self, result_type: _Optional[_Union[ResultType, str]] = ..., columns: _Optional[_Iterable[_Union[ColumnDescriptor, _Mapping]]] = ..., ordered: bool = ...) -> None: ...

class ColumnDescriptor(_message.Message):
    __slots__ = ("name", "type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: _gql_types_pb2.TypeDescriptor
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[_gql_types_pb2.TypeDescriptor, _Mapping]] = ...) -> None: ...

class RowBatch(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[Row]
    def __init__(self, rows: _Optional[_Iterable[_Union[Row, _Mapping]]] = ...) -> None: ...

class Row(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[_gql_types_pb2.Value]
    def __init__(self, values: _Optional[_Iterable[_Union[_gql_types_pb2.Value, _Mapping]]] = ...) -> None: ...

class ResultSummary(_message.Message):
    __slots__ = ("status", "warnings", "rows_affected", "counters")
    class CountersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    ROWS_AFFECTED_FIELD_NUMBER: _ClassVar[int]
    COUNTERS_FIELD_NUMBER: _ClassVar[int]
    status: _gql_types_pb2.GqlStatus
    warnings: _containers.RepeatedCompositeFieldContainer[_gql_types_pb2.GqlStatus]
    rows_affected: int
    counters: _containers.ScalarMap[str, int]
    def __init__(self, status: _Optional[_Union[_gql_types_pb2.GqlStatus, _Mapping]] = ..., warnings: _Optional[_Iterable[_Union[_gql_types_pb2.GqlStatus, _Mapping]]] = ..., rows_affected: _Optional[int] = ..., counters: _Optional[_Mapping[str, int]] = ...) -> None: ...

class BeginRequest(_message.Message):
    __slots__ = ("session_id", "mode")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    mode: TransactionMode
    def __init__(self, session_id: _Optional[str] = ..., mode: _Optional[_Union[TransactionMode, str]] = ...) -> None: ...

class BeginResponse(_message.Message):
    __slots__ = ("transaction_id", "status")
    TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    transaction_id: str
    status: _gql_types_pb2.GqlStatus
    def __init__(self, transaction_id: _Optional[str] = ..., status: _Optional[_Union[_gql_types_pb2.GqlStatus, _Mapping]] = ...) -> None: ...

class CommitRequest(_message.Message):
    __slots__ = ("session_id", "transaction_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    transaction_id: str
    def __init__(self, session_id: _Optional[str] = ..., transaction_id: _Optional[str] = ...) -> None: ...

class CommitResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: _gql_types_pb2.GqlStatus
    def __init__(self, status: _Optional[_Union[_gql_types_pb2.GqlStatus, _Mapping]] = ...) -> None: ...

class RollbackRequest(_message.Message):
    __slots__ = ("session_id", "transaction_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    transaction_id: str
    def __init__(self, session_id: _Optional[str] = ..., transaction_id: _Optional[str] = ...) -> None: ...

class RollbackResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: _gql_types_pb2.GqlStatus
    def __init__(self, status: _Optional[_Union[_gql_types_pb2.GqlStatus, _Mapping]] = ...) -> None: ...

class ListSchemasRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SchemaInfo(_message.Message):
    __slots__ = ("name", "graph_count", "graph_type_count")
    NAME_FIELD_NUMBER: _ClassVar[int]
    GRAPH_COUNT_FIELD_NUMBER: _ClassVar[int]
    GRAPH_TYPE_COUNT_FIELD_NUMBER: _ClassVar[int]
    name: str
    graph_count: int
    graph_type_count: int
    def __init__(self, name: _Optional[str] = ..., graph_count: _Optional[int] = ..., graph_type_count: _Optional[int] = ...) -> None: ...

class ListSchemasResponse(_message.Message):
    __slots__ = ("schemas",)
    SCHEMAS_FIELD_NUMBER: _ClassVar[int]
    schemas: _containers.RepeatedCompositeFieldContainer[SchemaInfo]
    def __init__(self, schemas: _Optional[_Iterable[_Union[SchemaInfo, _Mapping]]] = ...) -> None: ...

class CreateSchemaRequest(_message.Message):
    __slots__ = ("name", "if_not_exists")
    NAME_FIELD_NUMBER: _ClassVar[int]
    IF_NOT_EXISTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    if_not_exists: bool
    def __init__(self, name: _Optional[str] = ..., if_not_exists: bool = ...) -> None: ...

class CreateSchemaResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DropSchemaRequest(_message.Message):
    __slots__ = ("name", "if_exists")
    NAME_FIELD_NUMBER: _ClassVar[int]
    IF_EXISTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    if_exists: bool
    def __init__(self, name: _Optional[str] = ..., if_exists: bool = ...) -> None: ...

class DropSchemaResponse(_message.Message):
    __slots__ = ("existed",)
    EXISTED_FIELD_NUMBER: _ClassVar[int]
    existed: bool
    def __init__(self, existed: bool = ...) -> None: ...

class ListGraphsRequest(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: str
    def __init__(self, schema: _Optional[str] = ...) -> None: ...

class GraphSummary(_message.Message):
    __slots__ = ("schema", "name", "node_count", "edge_count", "graph_type")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    EDGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    GRAPH_TYPE_FIELD_NUMBER: _ClassVar[int]
    schema: str
    name: str
    node_count: int
    edge_count: int
    graph_type: str
    def __init__(self, schema: _Optional[str] = ..., name: _Optional[str] = ..., node_count: _Optional[int] = ..., edge_count: _Optional[int] = ..., graph_type: _Optional[str] = ...) -> None: ...

class ListGraphsResponse(_message.Message):
    __slots__ = ("graphs",)
    GRAPHS_FIELD_NUMBER: _ClassVar[int]
    graphs: _containers.RepeatedCompositeFieldContainer[GraphSummary]
    def __init__(self, graphs: _Optional[_Iterable[_Union[GraphSummary, _Mapping]]] = ...) -> None: ...

class CreateGraphRequest(_message.Message):
    __slots__ = ("schema", "name", "if_not_exists", "or_replace", "open_type", "graph_type_ref", "copy_of", "storage_mode", "options")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IF_NOT_EXISTS_FIELD_NUMBER: _ClassVar[int]
    OR_REPLACE_FIELD_NUMBER: _ClassVar[int]
    OPEN_TYPE_FIELD_NUMBER: _ClassVar[int]
    GRAPH_TYPE_REF_FIELD_NUMBER: _ClassVar[int]
    COPY_OF_FIELD_NUMBER: _ClassVar[int]
    STORAGE_MODE_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    schema: str
    name: str
    if_not_exists: bool
    or_replace: bool
    open_type: bool
    graph_type_ref: str
    copy_of: str
    storage_mode: str
    options: GraphOptions
    def __init__(self, schema: _Optional[str] = ..., name: _Optional[str] = ..., if_not_exists: bool = ..., or_replace: bool = ..., open_type: bool = ..., graph_type_ref: _Optional[str] = ..., copy_of: _Optional[str] = ..., storage_mode: _Optional[str] = ..., options: _Optional[_Union[GraphOptions, _Mapping]] = ...) -> None: ...

class GraphOptions(_message.Message):
    __slots__ = ("memory_limit_bytes", "backward_edges", "threads", "wal_enabled", "wal_durability")
    MEMORY_LIMIT_BYTES_FIELD_NUMBER: _ClassVar[int]
    BACKWARD_EDGES_FIELD_NUMBER: _ClassVar[int]
    THREADS_FIELD_NUMBER: _ClassVar[int]
    WAL_ENABLED_FIELD_NUMBER: _ClassVar[int]
    WAL_DURABILITY_FIELD_NUMBER: _ClassVar[int]
    memory_limit_bytes: int
    backward_edges: bool
    threads: int
    wal_enabled: bool
    wal_durability: str
    def __init__(self, memory_limit_bytes: _Optional[int] = ..., backward_edges: bool = ..., threads: _Optional[int] = ..., wal_enabled: bool = ..., wal_durability: _Optional[str] = ...) -> None: ...

class CreateGraphResponse(_message.Message):
    __slots__ = ("graph",)
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    graph: GraphSummary
    def __init__(self, graph: _Optional[_Union[GraphSummary, _Mapping]] = ...) -> None: ...

class DropGraphRequest(_message.Message):
    __slots__ = ("schema", "name", "if_exists")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IF_EXISTS_FIELD_NUMBER: _ClassVar[int]
    schema: str
    name: str
    if_exists: bool
    def __init__(self, schema: _Optional[str] = ..., name: _Optional[str] = ..., if_exists: bool = ...) -> None: ...

class DropGraphResponse(_message.Message):
    __slots__ = ("existed",)
    EXISTED_FIELD_NUMBER: _ClassVar[int]
    existed: bool
    def __init__(self, existed: bool = ...) -> None: ...

class GetGraphInfoRequest(_message.Message):
    __slots__ = ("schema", "name")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    schema: str
    name: str
    def __init__(self, schema: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class GetGraphInfoResponse(_message.Message):
    __slots__ = ("schema", "name", "node_count", "edge_count", "graph_type", "storage_mode", "memory_limit_bytes", "backward_edges", "threads")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    EDGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    GRAPH_TYPE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_MODE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_LIMIT_BYTES_FIELD_NUMBER: _ClassVar[int]
    BACKWARD_EDGES_FIELD_NUMBER: _ClassVar[int]
    THREADS_FIELD_NUMBER: _ClassVar[int]
    schema: str
    name: str
    node_count: int
    edge_count: int
    graph_type: str
    storage_mode: str
    memory_limit_bytes: int
    backward_edges: bool
    threads: int
    def __init__(self, schema: _Optional[str] = ..., name: _Optional[str] = ..., node_count: _Optional[int] = ..., edge_count: _Optional[int] = ..., graph_type: _Optional[str] = ..., storage_mode: _Optional[str] = ..., memory_limit_bytes: _Optional[int] = ..., backward_edges: bool = ..., threads: _Optional[int] = ...) -> None: ...

class ListGraphTypesRequest(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: str
    def __init__(self, schema: _Optional[str] = ...) -> None: ...

class GraphTypeInfo(_message.Message):
    __slots__ = ("schema", "name")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    schema: str
    name: str
    def __init__(self, schema: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ListGraphTypesResponse(_message.Message):
    __slots__ = ("graph_types",)
    GRAPH_TYPES_FIELD_NUMBER: _ClassVar[int]
    graph_types: _containers.RepeatedCompositeFieldContainer[GraphTypeInfo]
    def __init__(self, graph_types: _Optional[_Iterable[_Union[GraphTypeInfo, _Mapping]]] = ...) -> None: ...

class CreateGraphTypeRequest(_message.Message):
    __slots__ = ("schema", "name", "if_not_exists", "or_replace")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IF_NOT_EXISTS_FIELD_NUMBER: _ClassVar[int]
    OR_REPLACE_FIELD_NUMBER: _ClassVar[int]
    schema: str
    name: str
    if_not_exists: bool
    or_replace: bool
    def __init__(self, schema: _Optional[str] = ..., name: _Optional[str] = ..., if_not_exists: bool = ..., or_replace: bool = ...) -> None: ...

class CreateGraphTypeResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DropGraphTypeRequest(_message.Message):
    __slots__ = ("schema", "name", "if_exists")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IF_EXISTS_FIELD_NUMBER: _ClassVar[int]
    schema: str
    name: str
    if_exists: bool
    def __init__(self, schema: _Optional[str] = ..., name: _Optional[str] = ..., if_exists: bool = ...) -> None: ...

class DropGraphTypeResponse(_message.Message):
    __slots__ = ("existed",)
    EXISTED_FIELD_NUMBER: _ClassVar[int]
    existed: bool
    def __init__(self, existed: bool = ...) -> None: ...

class GetGraphStatsRequest(_message.Message):
    __slots__ = ("graph",)
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    graph: str
    def __init__(self, graph: _Optional[str] = ...) -> None: ...

class GetGraphStatsResponse(_message.Message):
    __slots__ = ("node_count", "edge_count", "label_count", "edge_type_count", "property_key_count", "index_count", "memory_bytes", "disk_bytes")
    NODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    EDGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    LABEL_COUNT_FIELD_NUMBER: _ClassVar[int]
    EDGE_TYPE_COUNT_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_KEY_COUNT_FIELD_NUMBER: _ClassVar[int]
    INDEX_COUNT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    node_count: int
    edge_count: int
    label_count: int
    edge_type_count: int
    property_key_count: int
    index_count: int
    memory_bytes: int
    disk_bytes: int
    def __init__(self, node_count: _Optional[int] = ..., edge_count: _Optional[int] = ..., label_count: _Optional[int] = ..., edge_type_count: _Optional[int] = ..., property_key_count: _Optional[int] = ..., index_count: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ...) -> None: ...

class WalStatusRequest(_message.Message):
    __slots__ = ("graph",)
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    graph: str
    def __init__(self, graph: _Optional[str] = ...) -> None: ...

class WalStatusResponse(_message.Message):
    __slots__ = ("enabled", "path", "size_bytes", "record_count", "last_checkpoint", "current_epoch")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    RECORD_COUNT_FIELD_NUMBER: _ClassVar[int]
    LAST_CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    CURRENT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    path: str
    size_bytes: int
    record_count: int
    last_checkpoint: int
    current_epoch: int
    def __init__(self, enabled: bool = ..., path: _Optional[str] = ..., size_bytes: _Optional[int] = ..., record_count: _Optional[int] = ..., last_checkpoint: _Optional[int] = ..., current_epoch: _Optional[int] = ...) -> None: ...

class WalCheckpointRequest(_message.Message):
    __slots__ = ("graph",)
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    graph: str
    def __init__(self, graph: _Optional[str] = ...) -> None: ...

class WalCheckpointResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ValidateRequest(_message.Message):
    __slots__ = ("graph",)
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    graph: str
    def __init__(self, graph: _Optional[str] = ...) -> None: ...

class ValidateResponse(_message.Message):
    __slots__ = ("valid", "errors", "warnings")
    VALID_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    valid: bool
    errors: _containers.RepeatedCompositeFieldContainer[ValidationError]
    warnings: _containers.RepeatedCompositeFieldContainer[ValidationWarning]
    def __init__(self, valid: bool = ..., errors: _Optional[_Iterable[_Union[ValidationError, _Mapping]]] = ..., warnings: _Optional[_Iterable[_Union[ValidationWarning, _Mapping]]] = ...) -> None: ...

class ValidationError(_message.Message):
    __slots__ = ("code", "message", "context")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    code: str
    message: str
    context: str
    def __init__(self, code: _Optional[str] = ..., message: _Optional[str] = ..., context: _Optional[str] = ...) -> None: ...

class ValidationWarning(_message.Message):
    __slots__ = ("code", "message", "context")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    code: str
    message: str
    context: str
    def __init__(self, code: _Optional[str] = ..., message: _Optional[str] = ..., context: _Optional[str] = ...) -> None: ...

class CreateIndexRequest(_message.Message):
    __slots__ = ("graph", "property_index", "vector_index", "text_index")
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_INDEX_FIELD_NUMBER: _ClassVar[int]
    VECTOR_INDEX_FIELD_NUMBER: _ClassVar[int]
    TEXT_INDEX_FIELD_NUMBER: _ClassVar[int]
    graph: str
    property_index: PropertyIndexDef
    vector_index: VectorIndexDef
    text_index: TextIndexDef
    def __init__(self, graph: _Optional[str] = ..., property_index: _Optional[_Union[PropertyIndexDef, _Mapping]] = ..., vector_index: _Optional[_Union[VectorIndexDef, _Mapping]] = ..., text_index: _Optional[_Union[TextIndexDef, _Mapping]] = ...) -> None: ...

class PropertyIndexDef(_message.Message):
    __slots__ = ("property",)
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    property: str
    def __init__(self, property: _Optional[str] = ...) -> None: ...

class VectorIndexDef(_message.Message):
    __slots__ = ("label", "property", "dimensions", "metric", "m", "ef_construction")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    M_FIELD_NUMBER: _ClassVar[int]
    EF_CONSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    label: str
    property: str
    dimensions: int
    metric: str
    m: int
    ef_construction: int
    def __init__(self, label: _Optional[str] = ..., property: _Optional[str] = ..., dimensions: _Optional[int] = ..., metric: _Optional[str] = ..., m: _Optional[int] = ..., ef_construction: _Optional[int] = ...) -> None: ...

class TextIndexDef(_message.Message):
    __slots__ = ("label", "property")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    label: str
    property: str
    def __init__(self, label: _Optional[str] = ..., property: _Optional[str] = ...) -> None: ...

class CreateIndexResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DropIndexRequest(_message.Message):
    __slots__ = ("graph", "property_index", "vector_index", "text_index")
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_INDEX_FIELD_NUMBER: _ClassVar[int]
    VECTOR_INDEX_FIELD_NUMBER: _ClassVar[int]
    TEXT_INDEX_FIELD_NUMBER: _ClassVar[int]
    graph: str
    property_index: PropertyIndexDef
    vector_index: VectorIndexDef
    text_index: TextIndexDef
    def __init__(self, graph: _Optional[str] = ..., property_index: _Optional[_Union[PropertyIndexDef, _Mapping]] = ..., vector_index: _Optional[_Union[VectorIndexDef, _Mapping]] = ..., text_index: _Optional[_Union[TextIndexDef, _Mapping]] = ...) -> None: ...

class DropIndexResponse(_message.Message):
    __slots__ = ("existed",)
    EXISTED_FIELD_NUMBER: _ClassVar[int]
    existed: bool
    def __init__(self, existed: bool = ...) -> None: ...

class VectorSearchRequest(_message.Message):
    __slots__ = ("graph", "label", "property", "query_vector", "k", "ef", "filters")
    class FiltersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _gql_types_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_gql_types_pb2.Value, _Mapping]] = ...) -> None: ...
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    EF_FIELD_NUMBER: _ClassVar[int]
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    graph: str
    label: str
    property: str
    query_vector: _containers.RepeatedScalarFieldContainer[float]
    k: int
    ef: int
    filters: _containers.MessageMap[str, _gql_types_pb2.Value]
    def __init__(self, graph: _Optional[str] = ..., label: _Optional[str] = ..., property: _Optional[str] = ..., query_vector: _Optional[_Iterable[float]] = ..., k: _Optional[int] = ..., ef: _Optional[int] = ..., filters: _Optional[_Mapping[str, _gql_types_pb2.Value]] = ...) -> None: ...

class TextSearchRequest(_message.Message):
    __slots__ = ("graph", "label", "property", "query", "k")
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    graph: str
    label: str
    property: str
    query: str
    k: int
    def __init__(self, graph: _Optional[str] = ..., label: _Optional[str] = ..., property: _Optional[str] = ..., query: _Optional[str] = ..., k: _Optional[int] = ...) -> None: ...

class HybridSearchRequest(_message.Message):
    __slots__ = ("graph", "label", "text_property", "vector_property", "query_text", "query_vector", "k")
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    TEXT_PROPERTY_FIELD_NUMBER: _ClassVar[int]
    VECTOR_PROPERTY_FIELD_NUMBER: _ClassVar[int]
    QUERY_TEXT_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    graph: str
    label: str
    text_property: str
    vector_property: str
    query_text: str
    query_vector: _containers.RepeatedScalarFieldContainer[float]
    k: int
    def __init__(self, graph: _Optional[str] = ..., label: _Optional[str] = ..., text_property: _Optional[str] = ..., vector_property: _Optional[str] = ..., query_text: _Optional[str] = ..., query_vector: _Optional[_Iterable[float]] = ..., k: _Optional[int] = ...) -> None: ...

class SearchHit(_message.Message):
    __slots__ = ("node_id", "score", "properties")
    class PropertiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _gql_types_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_gql_types_pb2.Value, _Mapping]] = ...) -> None: ...
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    node_id: int
    score: float
    properties: _containers.MessageMap[str, _gql_types_pb2.Value]
    def __init__(self, node_id: _Optional[int] = ..., score: _Optional[float] = ..., properties: _Optional[_Mapping[str, _gql_types_pb2.Value]] = ...) -> None: ...

class VectorSearchResponse(_message.Message):
    __slots__ = ("hits",)
    HITS_FIELD_NUMBER: _ClassVar[int]
    hits: _containers.RepeatedCompositeFieldContainer[SearchHit]
    def __init__(self, hits: _Optional[_Iterable[_Union[SearchHit, _Mapping]]] = ...) -> None: ...

class TextSearchResponse(_message.Message):
    __slots__ = ("hits",)
    HITS_FIELD_NUMBER: _ClassVar[int]
    hits: _containers.RepeatedCompositeFieldContainer[SearchHit]
    def __init__(self, hits: _Optional[_Iterable[_Union[SearchHit, _Mapping]]] = ...) -> None: ...

class HybridSearchResponse(_message.Message):
    __slots__ = ("hits",)
    HITS_FIELD_NUMBER: _ClassVar[int]
    hits: _containers.RepeatedCompositeFieldContainer[SearchHit]
    def __init__(self, hits: _Optional[_Iterable[_Union[SearchHit, _Mapping]]] = ...) -> None: ...
