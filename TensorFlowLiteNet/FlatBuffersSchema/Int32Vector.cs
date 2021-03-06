// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct Int32Vector : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static Int32Vector GetRootAsInt32Vector(ByteBuffer _bb) { return GetRootAsInt32Vector(_bb, new Int32Vector()); }
  public static Int32Vector GetRootAsInt32Vector(ByteBuffer _bb, Int32Vector obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public Int32Vector __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int Values(int j) { int o = __p.__offset(4); return o != 0 ? __p.bb.GetInt(__p.__vector(o) + j * 4) : (int)0; }
  public int ValuesLength { get { int o = __p.__offset(4); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<int> GetValuesBytes() { return __p.__vector_as_span<int>(4, 4); }
#else
  public ArraySegment<byte>? GetValuesBytes() { return __p.__vector_as_arraysegment(4); }
#endif
  public int[] GetValuesArray() { return __p.__vector_as_array<int>(4); }

  public static Offset<tflite.Int32Vector> CreateInt32Vector(FlatBufferBuilder builder,
      VectorOffset valuesOffset = default(VectorOffset)) {
    builder.StartTable(1);
    Int32Vector.AddValues(builder, valuesOffset);
    return Int32Vector.EndInt32Vector(builder);
  }

  public static void StartInt32Vector(FlatBufferBuilder builder) { builder.StartTable(1); }
  public static void AddValues(FlatBufferBuilder builder, VectorOffset valuesOffset) { builder.AddOffset(0, valuesOffset.Value, 0); }
  public static VectorOffset CreateValuesVector(FlatBufferBuilder builder, int[] data) { builder.StartVector(4, data.Length, 4); for (int i = data.Length - 1; i >= 0; i--) builder.AddInt(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateValuesVectorBlock(FlatBufferBuilder builder, int[] data) { builder.StartVector(4, data.Length, 4); builder.Add(data); return builder.EndVector(); }
  public static void StartValuesVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(4, numElems, 4); }
  public static Offset<tflite.Int32Vector> EndInt32Vector(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.Int32Vector>(o);
  }
};


}
