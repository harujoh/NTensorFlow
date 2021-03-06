// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct DimensionMetadata : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static DimensionMetadata GetRootAsDimensionMetadata(ByteBuffer _bb) { return GetRootAsDimensionMetadata(_bb, new DimensionMetadata()); }
  public static DimensionMetadata GetRootAsDimensionMetadata(ByteBuffer _bb, DimensionMetadata obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public DimensionMetadata __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public tflite.DimensionType Format { get { int o = __p.__offset(4); return o != 0 ? (tflite.DimensionType)__p.bb.GetSbyte(o + __p.bb_pos) : tflite.DimensionType.DENSE; } }
  public int DenseSize { get { int o = __p.__offset(6); return o != 0 ? __p.bb.GetInt(o + __p.bb_pos) : (int)0; } }
  public tflite.SparseIndexVector ArraySegmentsType { get { int o = __p.__offset(8); return o != 0 ? (tflite.SparseIndexVector)__p.bb.Get(o + __p.bb_pos) : tflite.SparseIndexVector.NONE; } }
  public TTable? ArraySegments<TTable>() where TTable : struct, IFlatbufferObject { int o = __p.__offset(10); return o != 0 ? (TTable?)__p.__union<TTable>(o + __p.bb_pos) : null; }
  public tflite.SparseIndexVector ArrayIndicesType { get { int o = __p.__offset(12); return o != 0 ? (tflite.SparseIndexVector)__p.bb.Get(o + __p.bb_pos) : tflite.SparseIndexVector.NONE; } }
  public TTable? ArrayIndices<TTable>() where TTable : struct, IFlatbufferObject { int o = __p.__offset(14); return o != 0 ? (TTable?)__p.__union<TTable>(o + __p.bb_pos) : null; }

  public static Offset<tflite.DimensionMetadata> CreateDimensionMetadata(FlatBufferBuilder builder,
      tflite.DimensionType format = tflite.DimensionType.DENSE,
      int dense_size = 0,
      tflite.SparseIndexVector array_segments_type = tflite.SparseIndexVector.NONE,
      int array_segmentsOffset = 0,
      tflite.SparseIndexVector array_indices_type = tflite.SparseIndexVector.NONE,
      int array_indicesOffset = 0) {
    builder.StartTable(6);
    DimensionMetadata.AddArrayIndices(builder, array_indicesOffset);
    DimensionMetadata.AddArraySegments(builder, array_segmentsOffset);
    DimensionMetadata.AddDenseSize(builder, dense_size);
    DimensionMetadata.AddArrayIndicesType(builder, array_indices_type);
    DimensionMetadata.AddArraySegmentsType(builder, array_segments_type);
    DimensionMetadata.AddFormat(builder, format);
    return DimensionMetadata.EndDimensionMetadata(builder);
  }

  public static void StartDimensionMetadata(FlatBufferBuilder builder) { builder.StartTable(6); }
  public static void AddFormat(FlatBufferBuilder builder, tflite.DimensionType format) { builder.AddSbyte(0, (sbyte)format, 0); }
  public static void AddDenseSize(FlatBufferBuilder builder, int denseSize) { builder.AddInt(1, denseSize, 0); }
  public static void AddArraySegmentsType(FlatBufferBuilder builder, tflite.SparseIndexVector arraySegmentsType) { builder.AddByte(2, (byte)arraySegmentsType, 0); }
  public static void AddArraySegments(FlatBufferBuilder builder, int arraySegmentsOffset) { builder.AddOffset(3, arraySegmentsOffset, 0); }
  public static void AddArrayIndicesType(FlatBufferBuilder builder, tflite.SparseIndexVector arrayIndicesType) { builder.AddByte(4, (byte)arrayIndicesType, 0); }
  public static void AddArrayIndices(FlatBufferBuilder builder, int arrayIndicesOffset) { builder.AddOffset(5, arrayIndicesOffset, 0); }
  public static Offset<tflite.DimensionMetadata> EndDimensionMetadata(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.DimensionMetadata>(o);
  }
};


}
