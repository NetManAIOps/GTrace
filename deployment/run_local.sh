cd anomaly_detection/src
GOMP_CPU_AFFINITY=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m tracegnn.models.gtrace.anomaly_detect_local
