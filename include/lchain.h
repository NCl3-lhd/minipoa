#ifndef LCHAIN_H
#define LCHAIN_H
#include "mmpriv.h"
#include <vector>

mm128_t* mg_lchain_dp(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg, size_t* _n, mm128_t* a, int* n_u_, mm128_t** _u, void* km);
int chain_dp(void* km, mm128_t* lchains, int n_lchains, mm128_v* anchors, int max_dist_x, int max_dist_y, int bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg);
void partition_windows(mm128_v* anchors, int tlen, int qlen, int max_drift, int max_win, std::vector<bool>* win_ab);
#endif