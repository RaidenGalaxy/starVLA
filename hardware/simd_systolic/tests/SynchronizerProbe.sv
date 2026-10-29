module SynchronizerProbe(input logic clock,externalReset,data,
 output logic releasedReset,toggle);
 SimdResetSynchronizer rst(.clock(clock),.externalReset(externalReset),.value(releasedReset));
 SimdToggleSynchronizer bit_sync(.clock(clock),.reset(externalReset),.data(data),.value(toggle));
endmodule
