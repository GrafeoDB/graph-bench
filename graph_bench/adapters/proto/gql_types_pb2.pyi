from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DurationQualifier(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DURATION_UNSPECIFIED: _ClassVar[DurationQualifier]
    DURATION_YEAR_TO_MONTH: _ClassVar[DurationQualifier]
    DURATION_DAY_TO_SECOND: _ClassVar[DurationQualifier]

class GqlType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TYPE_UNKNOWN: _ClassVar[GqlType]
    TYPE_NULL: _ClassVar[GqlType]
    TYPE_BOOLEAN: _ClassVar[GqlType]
    TYPE_EMPTY: _ClassVar[GqlType]
    TYPE_INT8: _ClassVar[GqlType]
    TYPE_INT16: _ClassVar[GqlType]
    TYPE_INT32: _ClassVar[GqlType]
    TYPE_INT64: _ClassVar[GqlType]
    TYPE_INT128: _ClassVar[GqlType]
    TYPE_INT256: _ClassVar[GqlType]
    TYPE_UINT8: _ClassVar[GqlType]
    TYPE_UINT16: _ClassVar[GqlType]
    TYPE_UINT32: _ClassVar[GqlType]
    TYPE_UINT64: _ClassVar[GqlType]
    TYPE_UINT128: _ClassVar[GqlType]
    TYPE_UINT256: _ClassVar[GqlType]
    TYPE_FLOAT16: _ClassVar[GqlType]
    TYPE_FLOAT32: _ClassVar[GqlType]
    TYPE_FLOAT64: _ClassVar[GqlType]
    TYPE_FLOAT128: _ClassVar[GqlType]
    TYPE_FLOAT256: _ClassVar[GqlType]
    TYPE_DECIMAL: _ClassVar[GqlType]
    TYPE_STRING: _ClassVar[GqlType]
    TYPE_BYTES: _ClassVar[GqlType]
    TYPE_DATE: _ClassVar[GqlType]
    TYPE_LOCAL_TIME: _ClassVar[GqlType]
    TYPE_ZONED_TIME: _ClassVar[GqlType]
    TYPE_LOCAL_DATETIME: _ClassVar[GqlType]
    TYPE_ZONED_DATETIME: _ClassVar[GqlType]
    TYPE_DURATION: _ClassVar[GqlType]
    TYPE_YEAR_MONTH_DURATION: _ClassVar[GqlType]
    TYPE_DAY_TIME_DURATION: _ClassVar[GqlType]
    TYPE_LIST: _ClassVar[GqlType]
    TYPE_RECORD: _ClassVar[GqlType]
    TYPE_PATH: _ClassVar[GqlType]
    TYPE_NODE: _ClassVar[GqlType]
    TYPE_EDGE: _ClassVar[GqlType]
    TYPE_NODE_REFERENCE: _ClassVar[GqlType]
    TYPE_EDGE_REFERENCE: _ClassVar[GqlType]
    TYPE_GRAPH_REFERENCE: _ClassVar[GqlType]
    TYPE_BINDING_TABLE_REFERENCE: _ClassVar[GqlType]
    TYPE_ANY: _ClassVar[GqlType]
    TYPE_PROPERTY_VALUE: _ClassVar[GqlType]
DURATION_UNSPECIFIED: DurationQualifier
DURATION_YEAR_TO_MONTH: DurationQualifier
DURATION_DAY_TO_SECOND: DurationQualifier
TYPE_UNKNOWN: GqlType
TYPE_NULL: GqlType
TYPE_BOOLEAN: GqlType
TYPE_EMPTY: GqlType
TYPE_INT8: GqlType
TYPE_INT16: GqlType
TYPE_INT32: GqlType
TYPE_INT64: GqlType
TYPE_INT128: GqlType
TYPE_INT256: GqlType
TYPE_UINT8: GqlType
TYPE_UINT16: GqlType
TYPE_UINT32: GqlType
TYPE_UINT64: GqlType
TYPE_UINT128: GqlType
TYPE_UINT256: GqlType
TYPE_FLOAT16: GqlType
TYPE_FLOAT32: GqlType
TYPE_FLOAT64: GqlType
TYPE_FLOAT128: GqlType
TYPE_FLOAT256: GqlType
TYPE_DECIMAL: GqlType
TYPE_STRING: GqlType
TYPE_BYTES: GqlType
TYPE_DATE: GqlType
TYPE_LOCAL_TIME: GqlType
TYPE_ZONED_TIME: GqlType
TYPE_LOCAL_DATETIME: GqlType
TYPE_ZONED_DATETIME: GqlType
TYPE_DURATION: GqlType
TYPE_YEAR_MONTH_DURATION: GqlType
TYPE_DAY_TIME_DURATION: GqlType
TYPE_LIST: GqlType
TYPE_RECORD: GqlType
TYPE_PATH: GqlType
TYPE_NODE: GqlType
TYPE_EDGE: GqlType
TYPE_NODE_REFERENCE: GqlType
TYPE_EDGE_REFERENCE: GqlType
TYPE_GRAPH_REFERENCE: GqlType
TYPE_BINDING_TABLE_REFERENCE: GqlType
TYPE_ANY: GqlType
TYPE_PROPERTY_VALUE: GqlType

class Value(_message.Message):
    __slots__ = ("null_value", "boolean_value", "integer_value", "unsigned_integer_value", "big_integer_value", "float_value", "big_float_value", "decimal_value", "string_value", "bytes_value", "date_value", "local_time_value", "zoned_time_value", "local_datetime_value", "zoned_datetime_value", "duration_value", "list_value", "record_value", "node_value", "edge_value", "path_value")
    NULL_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOLEAN_VALUE_FIELD_NUMBER: _ClassVar[int]
    INTEGER_VALUE_FIELD_NUMBER: _ClassVar[int]
    UNSIGNED_INTEGER_VALUE_FIELD_NUMBER: _ClassVar[int]
    BIG_INTEGER_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    BIG_FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DECIMAL_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    BYTES_VALUE_FIELD_NUMBER: _ClassVar[int]
    DATE_VALUE_FIELD_NUMBER: _ClassVar[int]
    LOCAL_TIME_VALUE_FIELD_NUMBER: _ClassVar[int]
    ZONED_TIME_VALUE_FIELD_NUMBER: _ClassVar[int]
    LOCAL_DATETIME_VALUE_FIELD_NUMBER: _ClassVar[int]
    ZONED_DATETIME_VALUE_FIELD_NUMBER: _ClassVar[int]
    DURATION_VALUE_FIELD_NUMBER: _ClassVar[int]
    LIST_VALUE_FIELD_NUMBER: _ClassVar[int]
    RECORD_VALUE_FIELD_NUMBER: _ClassVar[int]
    NODE_VALUE_FIELD_NUMBER: _ClassVar[int]
    EDGE_VALUE_FIELD_NUMBER: _ClassVar[int]
    PATH_VALUE_FIELD_NUMBER: _ClassVar[int]
    null_value: NullValue
    boolean_value: bool
    integer_value: int
    unsigned_integer_value: int
    big_integer_value: BigInteger
    float_value: float
    big_float_value: BigFloat
    decimal_value: Decimal
    string_value: str
    bytes_value: bytes
    date_value: Date
    local_time_value: LocalTime
    zoned_time_value: ZonedTime
    local_datetime_value: LocalDateTime
    zoned_datetime_value: ZonedDateTime
    duration_value: Duration
    list_value: GqlList
    record_value: Record
    node_value: Node
    edge_value: Edge
    path_value: Path
    def __init__(self, null_value: _Optional[_Union[NullValue, _Mapping]] = ..., boolean_value: bool = ..., integer_value: _Optional[int] = ..., unsigned_integer_value: _Optional[int] = ..., big_integer_value: _Optional[_Union[BigInteger, _Mapping]] = ..., float_value: _Optional[float] = ..., big_float_value: _Optional[_Union[BigFloat, _Mapping]] = ..., decimal_value: _Optional[_Union[Decimal, _Mapping]] = ..., string_value: _Optional[str] = ..., bytes_value: _Optional[bytes] = ..., date_value: _Optional[_Union[Date, _Mapping]] = ..., local_time_value: _Optional[_Union[LocalTime, _Mapping]] = ..., zoned_time_value: _Optional[_Union[ZonedTime, _Mapping]] = ..., local_datetime_value: _Optional[_Union[LocalDateTime, _Mapping]] = ..., zoned_datetime_value: _Optional[_Union[ZonedDateTime, _Mapping]] = ..., duration_value: _Optional[_Union[Duration, _Mapping]] = ..., list_value: _Optional[_Union[GqlList, _Mapping]] = ..., record_value: _Optional[_Union[Record, _Mapping]] = ..., node_value: _Optional[_Union[Node, _Mapping]] = ..., edge_value: _Optional[_Union[Edge, _Mapping]] = ..., path_value: _Optional[_Union[Path, _Mapping]] = ...) -> None: ...

class NullValue(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BigInteger(_message.Message):
    __slots__ = ("value", "is_signed")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    IS_SIGNED_FIELD_NUMBER: _ClassVar[int]
    value: bytes
    is_signed: bool
    def __init__(self, value: _Optional[bytes] = ..., is_signed: bool = ...) -> None: ...

class BigFloat(_message.Message):
    __slots__ = ("value", "width")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    value: bytes
    width: int
    def __init__(self, value: _Optional[bytes] = ..., width: _Optional[int] = ...) -> None: ...

class Decimal(_message.Message):
    __slots__ = ("unscaled", "scale")
    UNSCALED_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    unscaled: bytes
    scale: int
    def __init__(self, unscaled: _Optional[bytes] = ..., scale: _Optional[int] = ...) -> None: ...

class Date(_message.Message):
    __slots__ = ("year", "month", "day")
    YEAR_FIELD_NUMBER: _ClassVar[int]
    MONTH_FIELD_NUMBER: _ClassVar[int]
    DAY_FIELD_NUMBER: _ClassVar[int]
    year: int
    month: int
    day: int
    def __init__(self, year: _Optional[int] = ..., month: _Optional[int] = ..., day: _Optional[int] = ...) -> None: ...

class LocalTime(_message.Message):
    __slots__ = ("hour", "minute", "second", "nanosecond")
    HOUR_FIELD_NUMBER: _ClassVar[int]
    MINUTE_FIELD_NUMBER: _ClassVar[int]
    SECOND_FIELD_NUMBER: _ClassVar[int]
    NANOSECOND_FIELD_NUMBER: _ClassVar[int]
    hour: int
    minute: int
    second: int
    nanosecond: int
    def __init__(self, hour: _Optional[int] = ..., minute: _Optional[int] = ..., second: _Optional[int] = ..., nanosecond: _Optional[int] = ...) -> None: ...

class ZonedTime(_message.Message):
    __slots__ = ("time", "offset_minutes")
    TIME_FIELD_NUMBER: _ClassVar[int]
    OFFSET_MINUTES_FIELD_NUMBER: _ClassVar[int]
    time: LocalTime
    offset_minutes: int
    def __init__(self, time: _Optional[_Union[LocalTime, _Mapping]] = ..., offset_minutes: _Optional[int] = ...) -> None: ...

class LocalDateTime(_message.Message):
    __slots__ = ("date", "time")
    DATE_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    date: Date
    time: LocalTime
    def __init__(self, date: _Optional[_Union[Date, _Mapping]] = ..., time: _Optional[_Union[LocalTime, _Mapping]] = ...) -> None: ...

class ZonedDateTime(_message.Message):
    __slots__ = ("date", "time", "offset_minutes")
    DATE_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    OFFSET_MINUTES_FIELD_NUMBER: _ClassVar[int]
    date: Date
    time: LocalTime
    offset_minutes: int
    def __init__(self, date: _Optional[_Union[Date, _Mapping]] = ..., time: _Optional[_Union[LocalTime, _Mapping]] = ..., offset_minutes: _Optional[int] = ...) -> None: ...

class Duration(_message.Message):
    __slots__ = ("months", "nanoseconds")
    MONTHS_FIELD_NUMBER: _ClassVar[int]
    NANOSECONDS_FIELD_NUMBER: _ClassVar[int]
    months: int
    nanoseconds: int
    def __init__(self, months: _Optional[int] = ..., nanoseconds: _Optional[int] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("id", "labels", "properties")
    class PropertiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    id: bytes
    labels: _containers.RepeatedScalarFieldContainer[str]
    properties: _containers.MessageMap[str, Value]
    def __init__(self, id: _Optional[bytes] = ..., labels: _Optional[_Iterable[str]] = ..., properties: _Optional[_Mapping[str, Value]] = ...) -> None: ...

class Edge(_message.Message):
    __slots__ = ("id", "labels", "source_node_id", "target_node_id", "undirected", "properties")
    class PropertiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    UNDIRECTED_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    id: bytes
    labels: _containers.RepeatedScalarFieldContainer[str]
    source_node_id: bytes
    target_node_id: bytes
    undirected: bool
    properties: _containers.MessageMap[str, Value]
    def __init__(self, id: _Optional[bytes] = ..., labels: _Optional[_Iterable[str]] = ..., source_node_id: _Optional[bytes] = ..., target_node_id: _Optional[bytes] = ..., undirected: bool = ..., properties: _Optional[_Mapping[str, Value]] = ...) -> None: ...

class Path(_message.Message):
    __slots__ = ("nodes", "edges")
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    edges: _containers.RepeatedCompositeFieldContainer[Edge]
    def __init__(self, nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ...) -> None: ...

class GqlList(_message.Message):
    __slots__ = ("elements",)
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedCompositeFieldContainer[Value]
    def __init__(self, elements: _Optional[_Iterable[_Union[Value, _Mapping]]] = ...) -> None: ...

class Record(_message.Message):
    __slots__ = ("fields",)
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedCompositeFieldContainer[Field]
    def __init__(self, fields: _Optional[_Iterable[_Union[Field, _Mapping]]] = ...) -> None: ...

class Field(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: Value
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...

class TypeDescriptor(_message.Message):
    __slots__ = ("type", "nullable", "element_type", "fields", "precision", "scale", "min_length", "max_length", "max_cardinality", "is_group", "is_open", "duration_qualifier", "component_types")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NULLABLE_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    MIN_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_CARDINALITY_FIELD_NUMBER: _ClassVar[int]
    IS_GROUP_FIELD_NUMBER: _ClassVar[int]
    IS_OPEN_FIELD_NUMBER: _ClassVar[int]
    DURATION_QUALIFIER_FIELD_NUMBER: _ClassVar[int]
    COMPONENT_TYPES_FIELD_NUMBER: _ClassVar[int]
    type: GqlType
    nullable: bool
    element_type: TypeDescriptor
    fields: _containers.RepeatedCompositeFieldContainer[FieldDescriptor]
    precision: int
    scale: int
    min_length: int
    max_length: int
    max_cardinality: int
    is_group: bool
    is_open: bool
    duration_qualifier: DurationQualifier
    component_types: _containers.RepeatedCompositeFieldContainer[TypeDescriptor]
    def __init__(self, type: _Optional[_Union[GqlType, str]] = ..., nullable: bool = ..., element_type: _Optional[_Union[TypeDescriptor, _Mapping]] = ..., fields: _Optional[_Iterable[_Union[FieldDescriptor, _Mapping]]] = ..., precision: _Optional[int] = ..., scale: _Optional[int] = ..., min_length: _Optional[int] = ..., max_length: _Optional[int] = ..., max_cardinality: _Optional[int] = ..., is_group: bool = ..., is_open: bool = ..., duration_qualifier: _Optional[_Union[DurationQualifier, str]] = ..., component_types: _Optional[_Iterable[_Union[TypeDescriptor, _Mapping]]] = ...) -> None: ...

class FieldDescriptor(_message.Message):
    __slots__ = ("name", "type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: TypeDescriptor
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[TypeDescriptor, _Mapping]] = ...) -> None: ...

class GqlStatus(_message.Message):
    __slots__ = ("code", "message", "diagnostic", "cause")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DIAGNOSTIC_FIELD_NUMBER: _ClassVar[int]
    CAUSE_FIELD_NUMBER: _ClassVar[int]
    code: str
    message: str
    diagnostic: DiagnosticRecord
    cause: GqlStatus
    def __init__(self, code: _Optional[str] = ..., message: _Optional[str] = ..., diagnostic: _Optional[_Union[DiagnosticRecord, _Mapping]] = ..., cause: _Optional[_Union[GqlStatus, _Mapping]] = ...) -> None: ...

class DiagnosticRecord(_message.Message):
    __slots__ = ("operation", "operation_code", "current_schema", "invalid_reference")
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    OPERATION_CODE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    INVALID_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    operation: str
    operation_code: int
    current_schema: str
    invalid_reference: str
    def __init__(self, operation: _Optional[str] = ..., operation_code: _Optional[int] = ..., current_schema: _Optional[str] = ..., invalid_reference: _Optional[str] = ...) -> None: ...

class AuthCredentials(_message.Message):
    __slots__ = ("bearer_token", "basic")
    BEARER_TOKEN_FIELD_NUMBER: _ClassVar[int]
    BASIC_FIELD_NUMBER: _ClassVar[int]
    bearer_token: str
    basic: BasicAuth
    def __init__(self, bearer_token: _Optional[str] = ..., basic: _Optional[_Union[BasicAuth, _Mapping]] = ...) -> None: ...

class BasicAuth(_message.Message):
    __slots__ = ("username", "password")
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    PASSWORD_FIELD_NUMBER: _ClassVar[int]
    username: str
    password: str
    def __init__(self, username: _Optional[str] = ..., password: _Optional[str] = ...) -> None: ...
