// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct SequenceRNNOptions : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static SequenceRNNOptions GetRootAsSequenceRNNOptions(ByteBuffer _bb) { return GetRootAsSequenceRNNOptions(_bb, new SequenceRNNOptions()); }
  public static SequenceRNNOptions GetRootAsSequenceRNNOptions(ByteBuffer _bb, SequenceRNNOptions obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public SequenceRNNOptions __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public bool TimeMajor { get { int o = __p.__offset(4); return o != 0 ? 0!=__p.bb.Get(o + __p.bb_pos) : (bool)false; } }
  public tflite.ActivationFunctionType FusedActivationFunction { get { int o = __p.__offset(6); return o != 0 ? (tflite.ActivationFunctionType)__p.bb.GetSbyte(o + __p.bb_pos) : tflite.ActivationFunctionType.NONE; } }
  public bool AsymmetricQuantizeInputs { get { int o = __p.__offset(8); return o != 0 ? 0!=__p.bb.Get(o + __p.bb_pos) : (bool)false; } }

  public static Offset<tflite.SequenceRNNOptions> CreateSequenceRNNOptions(FlatBufferBuilder builder,
      bool time_major = false,
      tflite.ActivationFunctionType fused_activation_function = tflite.ActivationFunctionType.NONE,
      bool asymmetric_quantize_inputs = false) {
    builder.StartTable(3);
    SequenceRNNOptions.AddAsymmetricQuantizeInputs(builder, asymmetric_quantize_inputs);
    SequenceRNNOptions.AddFusedActivationFunction(builder, fused_activation_function);
    SequenceRNNOptions.AddTimeMajor(builder, time_major);
    return SequenceRNNOptions.EndSequenceRNNOptions(builder);
  }

  public static void StartSequenceRNNOptions(FlatBufferBuilder builder) { builder.StartTable(3); }
  public static void AddTimeMajor(FlatBufferBuilder builder, bool timeMajor) { builder.AddBool(0, timeMajor, false); }
  public static void AddFusedActivationFunction(FlatBufferBuilder builder, tflite.ActivationFunctionType fusedActivationFunction) { builder.AddSbyte(1, (sbyte)fusedActivationFunction, 0); }
  public static void AddAsymmetricQuantizeInputs(FlatBufferBuilder builder, bool asymmetricQuantizeInputs) { builder.AddBool(2, asymmetricQuantizeInputs, false); }
  public static Offset<tflite.SequenceRNNOptions> EndSequenceRNNOptions(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.SequenceRNNOptions>(o);
  }
};


}
