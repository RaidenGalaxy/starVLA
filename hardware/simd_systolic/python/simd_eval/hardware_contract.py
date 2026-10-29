"""Explicit logical interfaces; runtime clocks are deliberately separate."""
from .reference import Format, WIDTHS, ACC_WIDTHS
from .architecture import ARCH


def hardware_contract(caps):
    target = caps['target']
    storage = target in ('system', 'gemm')
    array = target in ('array', 'system', 'gemm')
    arithmetic = target in ('alu','pe','array','system','gemm')
    has_pe = target in ('pe','array','system','gemm')
    rows, cols = caps['rows'], caps['cols']
    formats = []
    for fmt in Format:
        width, acc = WIDTHS[fmt], ACC_WIDTHS[fmt]
        formats.append(dict(format=fmt.name, input_bits=width, lanes=64 // width,
                            signed_integer=fmt < Format.F4,
                            full_integer_product_bits=2 * width if fmt < Format.F4 else None,
                            accumulator_bits=acc, result_bits=(64 // width) * acc,
                            native_result_bytes_per_element=acc // 8,
                            implementation=('independent_integer_lanes' if fmt < Format.F4 else
                                            'FP64_FMA' if fmt == Format.F64 else
                                            'exact_expand_then_FP32_FMA')))
    result = dict(schema_version=1, operand_word_bits=64 if arithmetic else None, result_container_bits=512 if arithmetic else None,
                  formats=formats if arithmetic else [], k_max=65536 if has_pe else None, contexts=16 if has_pe else 0,
                  accumulator_rule='max(32, 2*input_bits + ceil(log2(65536))) for signed integers',
                  array_a_ports=rows if array else int(arithmetic), array_b_ports=cols if array else int(arithmetic),
                  array_operand_port_bits=64, array_result_read_ports=1 if array else 0,
                  array_result_read_bits=512 if array else 0,
                  contains_pe_context_state=has_pe,
                  context_logical_read_ports=2 if has_pe else 0, context_effective_write_ports=1 if has_pe else 0,
                  context_port_bits=512 if has_pe else None, seed_and_commit_writes_mutually_exclusive=True if has_pe else None,
                  process='ASAP7', process_corner='TT', voltage_V=0.70,
                  temperature_C=0, process_manufacturable=False,
                  storage_implementation='asap7_sram_macro' if storage else 'registers', sram_macro_used=storage,
                  architecture_sha256=ARCH['architecture_sha256'], sram_physical_certified=False,
                  physical_frequency_certified=False)
    if storage:
        result.update(storage_banks=[
            dict(name=name, bytes=size, word_bits=bits, read_ports=1, write_ports=1)
            for name, size, bits in [('A', 16384, 128), ('B', 32768, 128),
                                     ('C', 16384, 128), ('aggregate', 65536, 64)]],
            dma_read_data_ports=1, dma_write_data_ports=1, dma_data_bits=128,
            dma_write_byte_enable_bits=16, logical_transaction_bytes=64,
            standalone_read_dma_max_outstanding=32,
            active_read_max_outstanding=1 if target == 'gemm' else 32,
            write_dma_max_outstanding=1,
            active_read_controller='single_cached_line_gather' if target == 'gemm' else 'ReadDma',
            full_duplex_throughput_guaranteed=False,
            sram_single_port=True,partial_writes='atomic_bank_read_modify_write',
            sram_reset_clears_contents=False,either_external_reset_aborts_epoch=True,
            writeback_clock_domain='memory', auxiliary_alu_clock_domain='core',
            fifo_directions=dict(waves='memory_to_core', reads='memory_to_core',
                                 results='core_to_memory', dones='core_to_memory',
                                 auxRequests='memory_to_core', auxResults='core_to_memory',
                                 seeds='memory_to_core', seedAcks='core_to_memory'))
    return result
