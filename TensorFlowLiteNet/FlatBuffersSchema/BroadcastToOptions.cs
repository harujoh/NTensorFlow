// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct BroadcastToOptions : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static BroadcastToOptions GetRootAsBroadcastToOptions(ByteBuffer _bb) { return GetRootAsBroadcastToOptions(_bb, new BroadcastToOptions()); }
  public static BroadcastToOptions GetRootAsBroadcastToOptions(ByteBuffer _bb, BroadcastToOptions obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public BroadcastToOptions __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }


  public static void StartBroadcastToOptions(FlatBufferBuilder builder) { builder.StartTable(0); }
  public static Offset<tflite.BroadcastToOptions> EndBroadcastToOptions(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.BroadcastToOptions>(o);
  }
};


}
