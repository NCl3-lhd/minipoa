#ifndef ALIGN_H
#define ALIGN_H
#include <vector>
#include "graph.h"
#include "parameter.h"
#include "result.h"
#include "mem_alloc_utils.h"
#include <stdio.h>
#include "minimizer.h"

// std::vector<res_t> POA(para_t* para, const graph& DAG, const std::string& seq);
// std::vector<res_t> POA_SIMD(para_t* para, const graph& DAG, const std::string& _seq, aligned_buff_t* mpool = nullptr);
// std::vector<res_t> POA_SIMD_ORIGIN(const para_t* para, const graph* DAG, const std::string& _seq, aligned_buff_t* mpool = nullptr);
// std::vector<res_t> abPOA(const para_t* para, const graph* DAG, const minimizer_t* mm, int rid, const std::string& _seq, aligned_buff_t* mpool = nullptr);
std::vector<res_t> poa(const para_t* para, const graph* DAG, int beg_id, int end_id, int qid, const char* qseq, int qlen, aligned_buff_t* mpool = nullptr, bool ab_band = true, int band_b = -1); // final version
std::vector<res_t> alignment(const para_t* para, graph* DAG, minimizer_t* mm, int rid, const char* qseq, int qlen, aligned_buff_t* mpool = nullptr);
#endif