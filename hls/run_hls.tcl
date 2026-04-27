# Vitis HLS 자동화 스크립트
# 실행: vitis_hls -f run_hls.tcl

set project_name "svm_inference"
set top_function  "svm_inference"
set part          "xc7z020clg400-1"   ;# PYNQ-Z2

# 프로젝트 생성
open_project $project_name
set_top $top_function

# 소스 추가
add_files svm_inference.cpp
add_files -tb tb_svm_inference.cpp

# 솔루션 생성
open_solution "solution1" -flow_target vivado
set_part $part
create_clock -period 10 -name default   ;# 100MHz

# C 시뮬레이션
csim_design

# HLS 합성
csynth_design

# RTL 검증 (선택)
# cosim_design

# IP Export
export_design -format ip_catalog -output "../pynq/svm_ip"

close_project
