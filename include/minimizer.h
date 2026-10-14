#ifndef MINIMIZER_H
#define MINIMIZER_H
#include "parameter.h"
#include "sequence.h"
#include "ksort.h"
#include "mmpriv.h"

void mm_sketch(void* km, const char* str, int len, int w, int k, uint32_t rid, int is_hpc, mm128_v* p);
std::vector<std::pair<int, int>> tr_intervals(const mm128_v* mm, int len, int k, int max_dist, int merge_gap);
struct minimizer_t {
  void* km;
  size_t seqs_size;
  int maxl;
  std::vector<int> len;
  int* mm_h;
  mm128_v mm_v, sorted_mm_v;
  mm128_v tmm_v, sorted_tmm_v, qmm_v, sorted_qmm_v;
  std::vector<int> max_sim; // is seq_id;
  std::vector<int> ord; // index is rid
  std::vector<int> rid_to_ord; // index is ord rid -> ord

  minimizer_t(para_t* para, const std::vector<seq_t>& seqs);
  ~minimizer_t();
  void init(para_t* para, const std::vector<seq_t>& seqs);
  int collect_mm(void* km, const std::vector<seq_t>& seqs, para_t* para);
  void get_guide_tree(para_t* para);
  mm128_t find_mm(int& idx, int rid, int tarPos) const;
  mm128_t find_mm(int rid, int tarPos) const;
  mm128_t match_mm(uint64_t mm_x, int rid) const;
  int collect_anchors(mm128_v* anchors, int qlen, int max_occ, const int* tcov, const int* qcov);
  int dp_chaining(const para_t* para, mm128_v* anchors, int tlen, int qlen, std::vector<bool>* win_ab);
  mm128_v collect_anchors_bytseq(const para_t* para, int qid, const char* qseq, int qlen, int tid, const char* tseq, int tlen, std::vector<bool>* win_ab);
  // int collect_anchors(void* km, mm128_v* anchors, mm128_v mm, int* mm_c, int tid, int qid, int qlen, int k);
};
// std::vector<sequence> readFile(const char* path);
#endif