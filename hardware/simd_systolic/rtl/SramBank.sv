// Single-port synchronous SRAM adapter. Partial writes lock the bank for a
// read/merge/write sequence. Reset cancels control only, never clears SRAM.
module SimdSramBank #(
  parameter integer WIDTH=128, DEPTH=1024
) (
  input clock, reset,
  input readEnable, input [12:0] readAddress,
  output readReady, output reg readValid, output [WIDTH-1:0] readData,
  input writeValid, output writeReady,
  input [12:0] writeAddress, input [WIDTH-1:0] writeData,
  input [WIDTH/8-1:0] writeStrobe,
  output busy, output reg writeDone,
  output [7:0] rmwReadBytes, output [7:0] writeBytes
);
  localparam integer WORDS=WIDTH/64, BANKS=DEPTH/1024;
  reg rmw;
  reg [12:0] pendingAddress, readAddressQ;
  reg [WIDTH-1:0] pendingData;
  reg [WIDTH/8-1:0] pendingStrobe;
  wire writeFire=writeValid && writeReady;
  wire readFire=readEnable && readReady;
  wire [WIDTH-1:0] macroData [0:BANKS-1];
  wire [WORDS-1:0] partial;
  wire [WORDS-1:0] physicalWrites;
  genvar b,w;
  for(w=0;w<WORDS;w=w+1) begin: masks
    assign partial[w]=(|writeStrobe[w*8+:8]) && !(&writeStrobe[w*8+:8]);
    assign physicalWrites[w]=(writeFire && (&writeStrobe[w*8+:8])) ||
      (rmw && (|pendingStrobe[w*8+:8]) && !(&pendingStrobe[w*8+:8]));
  end
  assign rmwReadBytes=writeFire ? ($countones(partial)*8) : 0;
  assign writeBytes=reset ? 0 : $countones(physicalWrites)*8;
  assign writeReady=!reset && !rmw;
  assign readReady=!reset && !rmw && !writeValid;
  assign busy=rmw;
  assign readData=macroData[readAddressQ/1024];
  for(b=0;b<BANKS;b=b+1) begin: depthBanks
    for(w=0;w<WORDS;w=w+1) begin: words
      wire activeWrite=writeFire && writeAddress/1024==b;
      wire activeRmw=rmw && pendingAddress/1024==b;
      wire [7:0] strobe=writeStrobe[w*8+:8];
      wire [7:0] heldStrobe=pendingStrobe[w*8+:8];
      wire needsRmw=(|heldStrobe) && !(&heldStrobe);
      wire doRead=(readFire && readAddress/1024==b) || (activeWrite && partial[w]);
      wire doWrite=(activeWrite && (&strobe)) || (activeRmw && needsRmw);
      wire [9:0] address=rmw ? pendingAddress[9:0] :
                             writeFire ? writeAddress[9:0] : readAddress[9:0];
      wire [63:0] oldData=macroData[b][w*64+:64];
      wire [63:0] merged;
      for(genvar byteIndex=0;byteIndex<8;byteIndex=byteIndex+1) begin: mergeBytes
        assign merged[byteIndex*8+:8]=heldStrobe[byteIndex] ?
          pendingData[w*64+byteIndex*8+:8] : oldData[byteIndex*8+:8];
      end
      srambank_256x4x64_6t122 macro (
        .clk(clock),.ADDRESS(address),.wd(rmw ? merged : writeData[w*64+:64]),
        .banksel(!reset && (doRead || doWrite)),.read(doRead && !reset),
        .write(doWrite && !reset),.dataout(macroData[b][w*64+:64])
      );
    end
  end
  always @(posedge clock) begin
    if(reset) begin rmw<=0;readValid<=0;writeDone<=0;readAddressQ<=0;end
    else begin
      readValid<=readFire;
      writeDone<=(writeFire && !(|partial)) || rmw;
      if(readFire) begin
        assert(readAddress<DEPTH);readAddressQ<=readAddress;
      end
      if(writeFire) begin
        assert(writeAddress<DEPTH);
        pendingAddress<=writeAddress;pendingData<=writeData;pendingStrobe<=writeStrobe;
        rmw<=|partial;
      end else if(rmw) rmw<=0;
    end
  end
endmodule
