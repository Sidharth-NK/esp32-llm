## Hardware baseline
- Chip: ESP32-D0WD-V3 rev 3.1, dual core, 160 MHz (240 max)
- PSRAM: none
- Free heap at boot: ~304 KB (min free: 304356 bytes)
- Flash: 4 MB (detected); app uses ~126 KB for hello_world
- Static DRAM: 180 KB total

## FP32
| Model | Precision | CPU | Ctx | tok/s | Free heap after load | KV cache |
|---|---|---|---|---|---|---|
| stories260K | fp32 | 240 MHz | 128 | 5.97 | 122,372 B | 163,840 B |

Weights are memory-mapped from flash (0 B heap). The KV cache is the RAM
limit, so context is capped at 128. Model file: download stories260K.bin
from Hugging Face (karpathy/tinyllamas) and flash it to 0x110000.