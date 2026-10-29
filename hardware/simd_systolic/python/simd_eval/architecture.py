"""One authenticated architecture contract; configured clocks are not signoff."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/architecture.properties'


def read_architecture(path=CONFIG):
    values = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.split('=', 1)
        if key.strip() in values:
            raise ValueError('duplicate architecture key: ' + key)
        values[key.strip()] = value.strip()
    for key in ('schema_version', 'stages', 'rows', 'cols', 'core_period_ps',
                'memory_period_ps', 'dram_period_ps', 'sram_read_latency',
                'sram_macro_depth', 'sram_macro_width', 'sram_macro_count', 'sram_capacity_bytes'):
        values[key] = int(values[key])
    if values['stages'] != 2 or values['process'] != 'asap7':
        raise ValueError('only the two-stage ASAP7 architecture is supported')
    if tuple(values[k] for k in ('core_period_ps', 'memory_period_ps', 'dram_period_ps')) != (1000, 2000, 1000):
        raise ValueError('acceptance clocks must be core=1GHz, memory=500MHz, DRAM=1ns')
    if values['sram_read_latency'] != 1 or values['sram_macro_depth'] * values['sram_macro_width'] * values['sram_macro_count'] != values['sram_capacity_bytes'] * 8:
        raise ValueError('invalid SRAM organization')
    values['architecture_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return values


ARCH = read_architecture()
CORE_PS = ARCH['core_period_ps']
MEMORY_PS = ARCH['memory_period_ps']
DRAM_PS = ARCH['dram_period_ps']


def require_f2(stages):
    if stages != ARCH['stages']:
        raise ValueError('only stages=2 is supported; historical HF builds cannot be reused')


def validate_dram(path):
    import configparser
    config = configparser.ConfigParser()
    config.read(path)
    if round(config.getfloat('timing', 'tCK') * 1000) != DRAM_PS:
        raise ValueError('DRAM tCK differs from architecture.properties')
