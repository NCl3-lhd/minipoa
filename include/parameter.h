#ifndef PARAMETER_H
#define PARAMETER_H
#include <vector>
#include <string>
struct para_t {
  // 成员声明
  int thread; 
  int m;std::vector<int> mat;std::string mat_fp;// score matrix  m is alphabet size
  int match, mismatch, gap_open1, gap_open2, gap_ext1, gap_ext2; int inf_min;
  int b, f;// adband
  bool ab_band;
  int k, mm_w;// k-mer
  float mm_filter_ratio;
  int max_occ; // 锚点 minimizer 在共识或 query 中的最大出现次数，0 不过滤
  int tr_dist; // 串联重复检测：同一 minimizer 再次出现的最大距离，0 不检测
  bool enable_seeding;
  int bw, poa_w;
  bool progressive_poa;
  bool isRNA;
  int result; // result format 0 consensus,1 rc-msa 2 gfa
  int verbose;  //verbose level (0-2). 0: none, 1: information, 2: debug
  std::string  inc_fp;
};
#endif