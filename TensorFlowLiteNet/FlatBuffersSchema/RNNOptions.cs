// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct RNNOptions : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static RNNOptions GetRootAsRNNOptions(ByteBuffer _bb) { return GetRootAsRNNOptions(_bb, new RNNOptions()); }
  public static RNNOptions GetRootAsRNNOptions(ByteBuffer _bb, RNNOptions obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public RNNOptions __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public tflite.ActivationFunctionType FusedActivationFunction { get { int o = __p.__offset(4); return o != 0 ? (tflite.ActivationFunctionType)__p.bb.GetSbyte(o + __p.bb_pos) : tflite.ActivationFunctionType.NONE; } }
  public bool AsymmetricQuantizeInputs { get { int o = __p.__offset(6); return o != 0 ? 0!=__p.bb.Get(o + __p.bb_pos) : (bool)false; } }

  public static Offset<tflite.RNNOptions> CreateRNNOptions(FlatBufferBuilder builder,
      tflite.ActivationFunctionType fused_activation_function = tflite.ActivationFunctionType.NONE,
      bool asymmetric_quantize_inputs = false) {
    builder.StartTable(2);
    RNNOptions.AddAsymmetricQuantizeInputs(builder, asymmetric_quantize_inputs);
    RNNOptions.AddFusedActivationFunction(builder, fused_activation_function);
    return RNNOptions.EndRNNOptions(builder);
  }

  public static void StartRNNOptions(FlatBufferBuilder builder) { builder.StartTable(2); }
  public static void AddFusedActivationFunction(FlatBufferBuilder builder, tflite.ActivationFunctionType fusedActivationFunction) { builder.AddSbyte(0, (sbyte)fusedActivationFunction, 0); }
  public static void AddAsymmetricQuantizeInputs(FlatBufferBuilder builder, bool asymmetricQuantizeInputs) { builder.AddBool(1, asymmetricQuantizeInputs, false); }
  public static Offset<tflite.RNNOptions> EndRNNOptions(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.RNNOptions>(o);
  }
};


}
