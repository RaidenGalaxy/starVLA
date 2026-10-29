PYTHON ?= python3
STAGES ?= 2
ROWS ?= 2
JOBS ?= 8
PERIOD_NS ?= $(shell PYTHONPATH=python $(PYTHON) -c 'from simd_eval.architecture import CORE_PS; print(CORE_PS/1000)')
ifneq ($(STAGES),2)
$(error Only STAGES=2 is supported)
endif
.PHONY: bootstrap alu pe array system gemm dma check smoke regression physical clean
bootstrap:
	$(PYTHON) scripts/bootstrap.py
alu:
	$(PYTHON) scripts/build.py --kind alu --stages $(STAGES) -j $(JOBS)
pe:
	$(PYTHON) scripts/build.py --kind pe --stages $(STAGES) --hierarchical -j $(JOBS)
	$(PYTHON) scripts/check_pe.py --stages $(STAGES)
	$(PYTHON) scripts/check_pe_state.py --build-dir .build/pe_$(STAGES)_1x1
	$(PYTHON) scripts/check_ii.py --stages $(STAGES)
ifneq ($(filter 8 12,$(STAGES)),)
	$(PYTHON) scripts/check_pe_reset.py --build-dir .build/pe_$(STAGES)_1x1
endif
array:
	$(PYTHON) scripts/build.py --kind array --stages $(STAGES) --rows $(ROWS) --hierarchical -j $(JOBS)
system:
	$(PYTHON) scripts/build.py --kind system --stages $(STAGES) --rows $(ROWS) --hierarchical -j $(JOBS)
gemm:
	$(PYTHON) scripts/build.py --kind gemm --stages $(STAGES) --rows $(ROWS) --hierarchical -j $(JOBS)
dma:
	$(PYTHON) scripts/build.py --kind dma -j $(JOBS)
	$(PYTHON) scripts/build.py --kind write_dma -j $(JOBS)
	$(PYTHON) scripts/check_dma.py
check:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v
smoke: check alu array system
	$(PYTHON) scripts/check_alu.py --stages $(STAGES) --stall
	$(PYTHON) scripts/check_array.py --stages $(STAGES) --rows $(ROWS) --cols $(ROWS) --pipelined-read
	$(PYTHON) scripts/check_array_cycles.py --build-dir .build/array_$(STAGES)_$(ROWS)x$(ROWS)
	$(PYTHON) scripts/check_array_events.py --build-dir .build/array_$(STAGES)_$(ROWS)x$(ROWS)
	$(PYTHON) scripts/check_auxiliary.py --stages $(STAGES) --dram
	$(PYTHON) scripts/run_gemm.py --format I4 --stages $(STAGES) --rows $(ROWS) --cols $(ROWS) --dram
regression:
	$(PYTHON) scripts/regression.py --jobs $(JOBS)
physical:
	$(PYTHON) physical/run.py --target array --stages $(STAGES) --rows $(ROWS) --period $(PERIOD_NS) --pe-hierarchy
clean:
	rm -rf .build target project/target

.PHONY: migration-check
migration-check:
	$(PYTHON) scripts/verify_migration.py --jobs $(JOBS)
