#include "minimizer.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include "sequence.h"
#include "kvec.h"
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include "lchain.h"
/* modified from lh3/minimap2/sketch.c */
/********** start *************/
static inline uint64_t hash64(uint64_t key, uint64_t mask)
{
  key = (~key + (key << 21)) & mask; // key = (key << 21) - key - 1;
  key = key ^ key >> 24;
  key = ((key + (key << 3)) + (key << 8)) & mask; // key * 265
  key = key ^ key >> 14;
  key = ((key + (key << 2)) + (key << 4)) & mask; // key * 21
  key = key ^ key >> 28;
  key = (key + (key << 31)) & mask;
  return key;
}

typedef struct { // a simplified version of kdq
  int front, count;
  int a[32];
} tiny_queue_t;

static inline void tq_push(tiny_queue_t* q, int x)
{
  q->a[((q->count++) + q->front) & 0x1f] = x;
}

static inline int tq_shift(tiny_queue_t* q)
{
  int x;
  if (q->count == 0) return -1;
  x = q->a[q->front++];
  q->front &= 0x1f;
  --q->count;
  return x;
}

/**
 * Find symmetric (w,k)-minimizers on a DNA sequence
 *
 * @param km     thread-local memory pool; using NULL falls back to malloc()
 * @param str    DNA sequence
 * @param len    length of $str
 * @param w      find a minimizer for every $w consecutive k-mers
 * @param k      k-mer size
 * @param rid    reference ID; will be copied to the output $p array
 * @param is_hpc homopolymer-compressed or not
 * @param p      minimizers
 *               p->a[i].x = kMer<<8 | kmerSpan
 *               p->a[i].y = rid<<32 | lastPos<<1 | strand
 *               where lastPos is the position of the last base of the i-th minimizer,
 *               and strand indicates whether the minimizer comes from the top or the bottom strand.
 *               Callers may want to set "p->n = 0"; otherwise results are appended to p
 */
void mm_sketch(void* km, const char* str, int len, int w, int k, uint32_t rid, int is_hpc, mm128_v* p)
{
  uint64_t shift1 = 2 * (k - 1), mask = (1ULL << 2 * k) - 1, kmer[2] = { 0,0 };
  int i, j, l, buf_pos, min_pos, kmer_span = 0;
  mm128_t buf[256], min = { UINT64_MAX, UINT64_MAX };
  tiny_queue_t tq;

  assert(len > 0 && (w > 0 && w < 256) && (k > 0 && k <= 28)); // 56 bits for k-mer; could use long k-mers, but 28 enough in practice
  memset(buf, 0xff, w * 16);
  memset(&tq, 0, sizeof(tiny_queue_t));
  kv_resize(mm128_t, km, *p, p->n + len / w);

  for (i = l = buf_pos = min_pos = 0; i < len; ++i) {
    int c = nt4_table[(uint8_t)str[i]];
    mm128_t info = { UINT64_MAX, UINT64_MAX };
    if (c < 4) { // not an ambiguous base
      int z;
      if (is_hpc) {
        int skip_len = 1;
        if (i + 1 < len && nt4_table[(uint8_t)str[i + 1]] == c) {
          for (skip_len = 2; i + skip_len < len; ++skip_len)
            if (nt4_table[(uint8_t)str[i + skip_len]] != c)
              break;
          i += skip_len - 1; // put $i at the end of the current homopolymer run
        }
        tq_push(&tq, skip_len);
        kmer_span += skip_len;
        if (tq.count > k) kmer_span -= tq_shift(&tq);
      }
      else kmer_span = l + 1 < k ? l + 1 : k;
      kmer[0] = (kmer[0] << 2 | c) & mask;           // forward k-mer
      // std::cerr << i << " " << kmer[0] << "\n";
      kmer[1] = (kmer[1] >> 2) | (3ULL ^ c) << shift1; // reverse k-mer

      // if (kmer[0] == kmer[1]) continue; // skip "symmetric k-mers" as we don't know it strand
      // z = kmer[0] < kmer[1] ? 0 : 1; // strand
      z = 0; // Do not take strand into account  

      ++l;
      if (l >= k && kmer_span < 256) {
        // std::cerr << rid << " " << z << " " << kmer[z] << " " << i << "\n";
        info.x = hash64(kmer[z], mask) << 8 | kmer_span;
        // std::cerr << info.x << "\n";
        info.y = (uint64_t)rid << 32 | (uint32_t)i << 1 | z;
      }
    }
    else l = 0, tq.count = tq.front = 0, kmer_span = 0;
    buf[buf_pos] = info; // need to do this here as appropriate buf_pos and buf[buf_pos] are needed below
    if (l == w + k - 1 && min.x != UINT64_MAX) { // special case for the first window - because identical k-mers are not stored yet
      for (j = buf_pos + 1; j < w; ++j)
        if (min.x == buf[j].x && buf[j].y != min.y) kv_push(mm128_t, km, *p, buf[j]);
      for (j = 0; j < buf_pos; ++j)
        if (min.x == buf[j].x && buf[j].y != min.y) kv_push(mm128_t, km, *p, buf[j]);
    }
    if (info.x <= min.x) { // a new minimum; then write the old min
      if (l >= w + k && min.x != UINT64_MAX) kv_push(mm128_t, km, *p, min);
      min = info, min_pos = buf_pos;
    }
    else if (buf_pos == min_pos) { // old min has moved outside the window
      if (l >= w + k - 1 && min.x != UINT64_MAX) kv_push(mm128_t, km, *p, min);
      for (j = buf_pos + 1, min.x = UINT64_MAX; j < w; ++j) // the two loops are necessary when there are identical k-mers
        if (min.x >= buf[j].x) min = buf[j], min_pos = j; // >= is important s.t. min is always the closest k-mer
      for (j = 0; j <= buf_pos; ++j)
        if (min.x >= buf[j].x) min = buf[j], min_pos = j;
      if (l >= w + k - 1 && min.x != UINT64_MAX) { // write identical k-mers
        for (j = buf_pos + 1; j < w; ++j) // these two loops make sure the output is sorted
          if (min.x == buf[j].x && min.y != buf[j].y) kv_push(mm128_t, km, *p, buf[j]);
        for (j = 0; j <= buf_pos; ++j)
          if (min.x == buf[j].x && min.y != buf[j].y) kv_push(mm128_t, km, *p, buf[j]);
      }
    }
    if (++buf_pos == w) buf_pos = 0;
  }
  if (min.x != UINT64_MAX)
    kv_push(mm128_t, km, *p, min);
}
/************ end *************/

/*
  串联重复区间：同一 minimizer 在 max_dist 内再次出现即视为周期 p（两次结束位置之差）的串联重复，
  区间取 [首次出现起点 - p, 再次出现终点 + p]，相距不超过 merge_gap 的区间合并。
  mm 需按位置排序（mm_sketch 的输出即是）；结果为按起点排序的半开区间，写入 iv。
  last、iv 由调用方持有以复用内存，进入时会被清空。
*/
void tr_intervals(const mm128_v* mm, int len, int k, int max_dist, int merge_gap,
                  std::unordered_map<uint64_t, int>& last, std::vector<std::pair<int, int>>& iv) {
  last.clear(); // minimizer -> 上一次出现的结束位置
  iv.clear();
  last.reserve(mm->n);
  for (size_t j = 0; j < mm->n; ++j) {
    int ej = (uint32_t)mm->a[j].y >> 1;
    auto it = last.find(mm->a[j].x);
    if (it != last.end()) {
      int p = ej - it->second;
      if (p > 0 && p <= max_dist) iv.emplace_back(std::max(0, ej - p - k + 1 - p), std::min(len, ej + 1 + p));
      it->second = ej;
    }
    else last.emplace(mm->a[j].x, ej);
  }
  std::sort(iv.begin(), iv.end());
  // 原地合并：iv[0, w) 为已合并的区间
  size_t w = 0;
  for (size_t i = 0; i < iv.size(); ++i) {
    if (w > 0 && iv[i].first <= iv[w - 1].second + merge_gap) iv[w - 1].second = std::max(iv[w - 1].second, iv[i].second);
    else iv[w++] = iv[i];
  }
  iv.resize(w);
}

// 串联重复区间的前缀和：cov[e + 1] - cov[s] > 0 表示 [s, e] 与区间相交
static void tr_cov(const mm128_v* mm, int len, int k, int max_dist, std::vector<int>& cov,
                   std::unordered_map<uint64_t, int>& last, std::vector<std::pair<int, int>>& iv) {
  cov.assign(len + 1, 0);
  tr_intervals(mm, len, k, max_dist, 50, last, iv);
  for (const auto& v : iv)
    for (int i = v.first; i < v.second; ++i) cov[i + 1] = 1;
  for (int i = 0; i < len; ++i) cov[i + 1] += cov[i];
}

/* modified from yangao07/abPOA/abpoa_seed.c */
#define sort_key_mm128x(a) ((a).x)
KRADIX_SORT_INIT(mm128x, mm128_t, sort_key_mm128x, 8)
/********** start *************/
int minimizer_t::collect_mm(void* km, const std::vector<seq_t>& seqs, para_t* para) {

  mm_h = (int*)malloc((seqs_size + 2) * sizeof(int));
  mm_h[0] = 0; // record mm i
  for (int i = 0; i < seqs.size(); ++i) { // collect minimizers
    // if (abpt->m > 5) mm_aa_sketch(km, seqs[i], seq_lens[i], abpt->w, abpt->k, i, 0, mm);
    // std::cerr << i << "\n";
    if (para->m <= 5) mm_sketch(km, seqs[i].seq.c_str(), seqs[i].seq.size(), para->mm_w, para->k, i, 0, &mm_v);
    mm_h[i + 1] = mm_v.n;
    if (para->verbose && i % 10 == 0) {
      std::cerr << "[" << i << "/" << seqs.size() << "]" << "\n";
      // std::cerr << seqs[i].seq.size() << " " << mm_h[i + 1] - mm_h[i] << "\n";
    }
  }
  return mm_v.n;
}
minimizer_t::minimizer_t(para_t* para, const std::vector<seq_t>& seqs) {
  init(para, seqs);
};

void minimizer_t::init(para_t* para, const std::vector<seq_t>& seqs) {
  km = nullptr;
  mm_v = { 0, 0, nullptr };
  sorted_mm_v = { 0, 0, nullptr };
  tmm_v = { 0, 0, nullptr };
  sorted_tmm_v = { 0, 0, nullptr };
  qmm_v = { 0, 0, nullptr };
  sorted_qmm_v = { 0, 0, nullptr };
  mm_h = nullptr;
  seqs_size = seqs.size();
  ord.resize(seqs_size);
  rid_to_ord.resize(seqs_size);
  std::iota(ord.begin(), ord.end(), 0);
  std::iota(rid_to_ord.begin(), rid_to_ord.end(), 0);
  if (para->progressive_poa) {
    km = km_init();
    collect_mm(km, seqs, para);//mm ->Minimizer
    maxl = 0;
    // std::cerr << "111 " << "\n";
    len.resize(seqs_size);
    for (int i = 0; i < seqs_size; i++) {
      len[i] = seqs[i].seq.size();
      maxl = std::max(maxl, len[i]);
    }
    // std::cerr << "collect_mm" << "\n";

    // if (para->enable_seeding) {
    //   for (int i = 0; i < (int)mm_v.n; ++i) kv_push(mm128_t, km, sorted_mm_v, mm_v.a[i]);
    //   for (int i = 0; i < seqs_size; i++) {
    //     radix_sort_mm128x(sorted_mm_v.a + mm_h[i], sorted_mm_v.a + mm_h[i + 1]); // sort mm by k-mer hash values
    //     // for (int j = mm_h[i]; j < mm_h[i + 1]; j++) {
    //     //   std::cerr << mm_v.a[j].x << " " << ((mm_v.a[j].y >> 1) & 0x7FFFFFFF) << " ";
    //     // }
    //     // std::cerr << "\n";
    //   }
    // }
    // std::cerr << "finish collect_mm" << "\n";
  }

};

void minimizer_t::get_guide_tree(para_t* para) {
  if (para->progressive_poa && seqs_size >= 2) {
    // copy mm1 to mm2
    mm128_v mm_tv = { 0, 0, nullptr };
    for (int i = 0; i < (int)mm_v.n; ++i) kv_push(mm128_t, km, mm_tv, mm_v.a[i]);
    // use mm2 to build guide tree
    if (mm_tv.n == 0) {
      // return ord;
      if (para->verbose) std::cerr << "no minimizer" << "\n";
      kfree(km, mm_tv.a);
      return;
    }

    // if (abpt->verbose >= ABPOA_INFO_VERBOSE) fprintf(stderr, "[%s] Building progressive guide tree ... ", __func__);
    size_t i, _i, j; int rid1, rid2;                                          // mm_hit_n: mimizer hits between each two sequences
    // 0: 0
    // 1: 0 1
    // 2: 0 1 2
    int* mm_hit_n = (int*)calloc((seqs_size * (seqs_size + 1)) >> 1, sizeof(int)); //  ...
    // n: 0 1 ... n-1 n
    // 
    // # total mimizers of i: mm_hit_n[(i*(i+1))/2+i]
    // # total hits for i and j (i>j): mm_hit_n[(i*(i+1)/2)+j]
    // std::cerr << mm_tv.n << "\n";
    radix_sort_mm128x(mm_tv.a, mm_tv.a + mm_tv.n); // sort mm by k-mer hash values

    uint64_t last_x = mm_tv.a[0].x;
    int *mm_cnt = (int *)calloc(seqs_size, sizeof(int));

    // 【优化1】：用 vector 记录当前 minimizer 到底存在于哪些序列中
    std::vector<int> present_rids;
    present_rids.reserve(seqs_size);

    // 注意：把条件放宽到 i <= mm_tv.n，利用尾部条件自动处理最后一块，省去你原来在循环外面复制的几十行代码
    for (_i = 0, i = 1; i <= mm_tv.n; ++i) {
      if (i == mm_tv.n || mm_tv.a[i].x != last_x) {
        present_rids.clear();
        for (j = _i; j < i; ++j) {
          rid1 = mm_tv.a[j].y >> 32;
          if (mm_cnt[rid1] == 0) present_rids.push_back(rid1);
          ++mm_cnt[rid1];
        }

        // -------------------------------------------------------------
        if (present_rids.size() > seqs_size * para->mm_filter_ratio) { 
            for (int r : present_rids) mm_cnt[r] = 0; // 重置清理
            if (i < mm_tv.n) { last_x = mm_tv.a[i].x; _i = i; }
            continue; 
        }

        for (size_t k1 = 0; k1 < present_rids.size(); ++k1) {
          rid1 = present_rids[k1];
          // 处理 self-hit
          mm_hit_n[((rid1 * (rid1 + 1)) >> 1) + rid1] += mm_cnt[rid1];

          for (size_t k2 = k1 + 1; k2 < present_rids.size(); ++k2) {
            rid2 = present_rids[k2];
            int max_r = std::max(rid1, rid2);
            int min_r = std::min(rid1, rid2);
            mm_hit_n[((max_r * (max_r + 1)) >> 1) + min_r] += std::min(mm_cnt[rid1], mm_cnt[rid2]);
          }
        }

        // 极速重置 mm_cnt，避免 memset 整个大数组
        for (int r : present_rids) mm_cnt[r] = 0;

        if (i < mm_tv.n) {
          last_x = mm_tv.a[i].x; _i = i;
        }
      }
    }
    free(mm_cnt);

    // calculate jaccard similarity between each two sequences
    double *jac_sim = (double *)calloc((seqs_size * seqs_size), sizeof(double));
    double max_jac = -1.0, jac; int max_i = -1, max_j = -1;
    for (i = 0; i < (size_t)seqs_size; ++i) {
      for (j = 0; j < (size_t)seqs_size; ++j) {
        if (i == j) continue;
        int shared_n = i > j ? mm_hit_n[((i * (i + 1)) >> 1) + j] : mm_hit_n[((j * (j + 1)) >> 1) + i];
        int tot_n = mm_hit_n[((i * (i + 1)) >> 1) + i] + mm_hit_n[((j * (j + 1)) >> 1) + j] - shared_n;
        if (tot_n == 0) jac = 0;
        else if (tot_n < 0) {
          std::cerr << __func__ << "Bug in progressive tree building. (1)" << "\n";
          exit(EXIT_FAILURE);
        }
        else {
          jac = (0.0 + shared_n) / tot_n;
          jac = jac * ((0.51 * len[i] / maxl) + (0.49 * len[j] / maxl));
        }
        jac_sim[i * seqs_size + j] = jac;
        if (jac > max_jac) {
          max_jac = jac; max_i = i, max_j = j;
        }
      }
    }

    max_sim.resize(seqs_size);
    std::vector<bool> in_ord(seqs_size, false);
    std::vector<double> sum_jac(seqs_size, 0.0);       // 记录未加入节点与已加入群体的相似度和
    std::vector<double> max_edge(seqs_size, -1.0);     // 记录未加入节点连接到树上的最大单边相似度
    std::vector<int> best_parent(seqs_size, -1);       // 记录未加入节点的最佳父节点

    int n_in_map = 2;
    ord[0] = max_i; ord[1] = max_j;
    rid_to_ord[max_i] = 0; rid_to_ord[max_j] = 1;
    max_sim[max_j] = max_i;
    in_ord[max_i] = true; in_ord[max_j] = true;

    // 初始化所有未加入节点的状态 (仅针对已放入树中的 max_i 和 max_j)
    for (int v = 0; v < seqs_size; ++v) {
      if (in_ord[v]) continue;
      sum_jac[v] = jac_sim[max_i * seqs_size + v] + jac_sim[max_j * seqs_size + v];

      if (jac_sim[max_i * seqs_size + v] >= jac_sim[max_j * seqs_size + v]) {
        max_edge[v] = jac_sim[max_i * seqs_size + v]; best_parent[v] = max_i;
      }
      else {
        max_edge[v] = jac_sim[max_j * seqs_size + v]; best_parent[v] = max_j;
      }
    }

    while (n_in_map < seqs_size) {
      // 1. O(N) 寻找当前 sum_jac 最大的未加入节点
      double best_sum = -1.0;
      int u = -1;
      for (int v = 0; v < seqs_size; ++v) {
        if (!in_ord[v] && sum_jac[v] > best_sum) {
          best_sum = sum_jac[v];
          u = v;
        }
      }

      // 处理特殊孤岛情况，防止死循环
      if (u == -1) {
        for (int v = 0; v < seqs_size; ++v) if (!in_ord[v]) { u = v; break; }
      }

      // 2. 将 u 加入树中
      in_ord[u] = true;
      ord[n_in_map] = u;
      rid_to_ord[u] = n_in_map;
      max_sim[u] = best_parent[u]; // 直接 O(1) 获取最佳父节点
      n_in_map++;

      // 3. O(N) 增量更新剩余节点的 sum_jac 和 best_parent (取代原本的 O(N^2) 扫描)
      for (int v = 0; v < seqs_size; ++v) {
        if (!in_ord[v]) {
          double current_jac = jac_sim[u * seqs_size + v];
          sum_jac[v] += current_jac;

          // 如果新加入的 u 到 v 的距离，比 v 之前记录的父节点更好，就换父节点
          if (current_jac > max_edge[v] || best_parent[v] == -1) {
            max_edge[v] = current_jac;
            best_parent[v] = u;
          }
        }
      }
    }

    free(mm_hit_n);
    free(jac_sim);
    // if (abpt->verbose >= ABPOA_INFO_VERBOSE) fprintf(stderr, "done!\n");

    kfree(km, mm_tv.a);
  }
  // return ord;
};

minimizer_t::~minimizer_t() {
  // std::cerr << "delete" << "\n";
  kfree(km, mm_v.a);kfree(km, sorted_mm_v.a); free(mm_h);
  kfree(km, tmm_v.a);kfree(km, sorted_tmm_v.a);
  kfree(km, qmm_v.a);kfree(km, sorted_qmm_v.a);
  // std::cerr << 1 << "\n";
  km_destroy(km);
  // std::cerr << 2 << "\n";

}
/*
  minimizers
  mm->a[i].x = kMer<<8 | kmerSpan
  mm->a[i].y = rid<<32 | lastPos<<1 | strand
  anchors
  a[].x: rev<<63 | tid<<32 | tpos
  a[].y: qid << 48 | flags<<40 | q_span<<32 | qpos
*/
int minimizer_t::collect_anchors(mm128_v* anchors, int qlen, int max_occ, const int* tcov, const int* qcov) {
  int i = 0, j = 0, _i, _j;
  while (i < sorted_tmm_v.n && j < sorted_qmm_v.n) {
    uint64_t xi = sorted_tmm_v.a[i].x, xj = sorted_qmm_v.a[j].x;
    if (xi == xj) {
      // 重复 minimizer（如串联重复内）可沿相差整数个重复单元的多条对角线配对，锚点位置有偶然性，直接跳过
      int i_end = i, j_end = j;
      while (i_end < sorted_tmm_v.n && sorted_tmm_v.a[i_end].x == xi) ++i_end;
      while (j_end < sorted_qmm_v.n && sorted_qmm_v.a[j_end].x == xj) ++j_end;
      if (max_occ > 0 && (i_end - i > max_occ || j_end - j > max_occ)) {
        i = i_end, j = j_end;
        continue;
      }
      int _i = i;
      for (; _i < sorted_tmm_v.n; ++_i) {
        uint64_t _xi = sorted_tmm_v.a[_i].x;
        if (_xi != xi) break;
        uint64_t _yi = sorted_tmm_v.a[_i].y;
        for (_j = j; _j < sorted_qmm_v.n; ++_j) {
          uint64_t _xj = sorted_qmm_v.a[_j].x;
          if (_xj != xj) break;
          uint64_t _yj = sorted_qmm_v.a[_j].y;
          // t_strand<<63 | t_lastPos<<32 | q_lastPos
          mm128_t anchor;
          uint32_t qspan = _xj & 0xff;
          // 与共识或 query 上的串联重复区间相交的锚点不生成，让分段跨过整个重复区
          int te = (uint32_t)_yi >> 1, qe = (uint32_t)_yj >> 1;
          if (tcov && tcov[te + 1] - tcov[te + 1 - qspan] > 0) continue;
          if (qcov && qcov[qe + 1] - qcov[qe + 1 - qspan] > 0) continue;
          if ((_yi & 1) == (_yj & 1)) { // same strand
            // p->x = (r & 0xffffffff00000000ULL) | rpos;
            // p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
            anchor.x = (_yi & 0x7fffffff00000000ULL) | ((uint32_t)_yi >> 1);
            anchor.y = (uint64_t)qspan << 32 | ((uint32_t)_yj >> 1);
          }
          else { // different strand
            anchor.x = 1ULL << 63 | (_yi & 0xffffffff00000000ULL) | ((uint32_t)_yi >> 1);
            anchor.y = (uint64_t)qspan << 32 | (qlen - (((uint32_t)_yj >> 1) + 1 - qspan) - 1);
          }
          kv_push(mm128_t, km, *anchors, anchor);
        }
      }
      i = _i, j = _j;
    }
    else if (xi < xj) ++i;
    else if (xj < xi) ++j;
  }
  // sort by tpos
  radix_sort_mm128x(anchors->a, anchors->a + anchors->n);
  return anchors->n;
}
/************ end *************/


mm128_t minimizer_t::find_mm(int rid, int tarPos) const {
  mm128_t res = { UINT64_MAX, UINT64_MAX };
  int l = mm_h[rid], r = mm_h[rid + 1];
  while (l + 1 < r) {
    int mid = (l + r) / 2;
    int curPos = (mm_v.a[mid].y >> 1) & 0x7FFFFFFF;
    if (curPos <= tarPos) {
      l = mid;
    }
    else {
      r = mid;
    }
  }
  return mm_v.a[l];
};

mm128_t minimizer_t::find_mm(int& idx, int rid, int tarPos) const {
  mm128_t res = { UINT64_MAX, UINT64_MAX };
  if (tarPos < 0) return res;
  while (idx < mm_h[rid + 1]) {
    int curPos = (mm_v.a[idx].y >> 1) & 0x7FFFFFFF;
    // std::cerr << curPos << " " << mm_v.a[idx].x << "\n";
    if (curPos > tarPos) break;
    if (curPos == tarPos) {
      res = mm_v.a[idx];
    }
    idx++;
  };
  return res;
};
mm128_t minimizer_t::match_mm(uint64_t mm_x, int rid) const {
  // if (mm_x == 2314759) {
  //   for 
  // }
  int l = mm_h[rid], r = mm_h[rid + 1];
  while (l + 1 < r) {
    int mid = (l + r) / 2;
    if (sorted_mm_v.a[mid].x <= mm_x) {
      l = mid;
    }
    else {
      r = mid;
    }
  }
  return sorted_mm_v.a[l];
};

int minimizer_t::dp_chaining(const para_t* para, mm128_v* anchors, int tlen, int qlen, std::vector<bool>* win_ab) {
  // mg_lchain_dp
  int max_bw = para->bw, max_dis = max_bw, max_skip_anchors = 25, max_non_best_anchors = 50, min_local_chain_cnt = 3, min_local_chain_score = 100;
  float chn_pen_gap = 0.01 * para->k, chn_pen_skip = 0.0 * 0.01 * para->k;
  int n_lchains;
  mm128_t* lchains;
  if (para->verbose >= 2) std::cerr << "initial anchor size:" << anchors->n << "\n";
  anchors->a = mg_lchain_dp(max_dis, max_dis, max_bw, max_skip_anchors, max_non_best_anchors, min_local_chain_cnt, min_local_chain_score, chn_pen_gap, chn_pen_skip, 0, 1, &anchors->n, anchors->a, &n_lchains, &lchains, km);
  if (para->verbose >= 2) std::cerr << "after mg_lchain_dp anchor size:" << anchors->n << "\n";
  if (para->verbose >= 2) std::cerr << "after mg_lchain_dp lchains number:" << n_lchains << "\n";
  // sort a by tpos
  // get a complete chain
  radix_sort_mm128x(lchains, lchains + n_lchains);
  // if (para->verbose) {
  //   // std::cerr << (int)(lchains[0].y >> 32) << " " << (int)lchains[0].y << "\n";
  //   for (int i = 0; i < n_lchains; i++) {
  //     uint64_t xi = lchains[i].x, yi = lchains[i].y;
  //     int st = (yi >> 32), ed = (int)yi;
  //     // std::cerr << "i:" << i << "\n";
  //     // std::cerr << "tpos:" << ((int)anchors->a[st].x) << " " << (xi >> 32) << " " << tlen << "\n";
  //     // std::cerr << "reg size:" << (xi >> 32) - ((int)anchors->a[st].x) << "\n";
  //     // std::cerr << "cnt:"<< ed - st << "\n";
  //     // std::cerr << "i [st, ed):"<< i << " " << st << " " << ed << "\n";
  //     // for (int i = st + 1; i < ed; i++) {
  //     //   std::cerr << ((int)anchors->a[i].x - (int)anchors->a[i - 1].x) << " ";
  //     // }
  //     std::cerr << "\n";
  //     // for (int i = st; i < ed; i++) {
  //     //   std::cerr << (int)anchors->a[i].y << " ";
  //     // }
  //     // std::cerr << "\n";
  //   }
  // }

  // find bug

  chain_dp(km, lchains, n_lchains, anchors, max_dis, max_dis, max_bw, chn_pen_gap, chn_pen_skip, 0, 1);
  if (para->verbose >= 2) std::cerr << "after chain_dp anchor size:" << anchors->n << "\n";
  // 最大窗口长度随查询序列长度缩放：max(500, qlen / 25)，且不超过 --win-max
  int max_win = std::min(para->win_max, std::max(500, qlen / 25));
  partition_windows(anchors, tlen, qlen, para->win_band, max_win, win_ab);
  if (para->verbose >= 2) {
    int n_ab = 0;
    for (bool ab : *win_ab) n_ab += ab;
    std::cerr << "windows:" << win_ab->size() << " adaptive:" << n_ab << "\n";
  }
  kfree(km, lchains);
  return 0;
}

mm128_v minimizer_t::collect_anchors_bytseq(const para_t* para, int qid, const char* qseq, int qlen, int tid, const char* tseq, int tlen, std::vector<bool>* win_ab) {
  // collect cons 's minimizer
  if (para->verbose >= 2) std::cerr << "collect tseq and "<< qid <<" qseq's minimizer" << "\n";
  int n = seqs_size; // cons 's rid
  tmm_v.n = 0;
  if (para->m <= 5) mm_sketch(km, tseq, tlen, para->mm_w, para->k, n, 0, &tmm_v);

  sorted_tmm_v.n = 0;
  for (int i = 0; i < (int)tmm_v.n; ++i) kv_push(mm128_t, km, sorted_tmm_v, tmm_v.a[i]);
  radix_sort_mm128x(sorted_tmm_v.a, sorted_tmm_v.a + tmm_v.n);

  qmm_v.n = 0;
  if (para->m <= 5) mm_sketch(km, qseq, qlen, para->mm_w, para->k, qid, 0, &qmm_v);

  sorted_qmm_v.n = 0;
  for (int i = 0; i < (int)qmm_v.n; ++i) kv_push(mm128_t, km, sorted_qmm_v, qmm_v.a[i]);
  radix_sort_mm128x(sorted_qmm_v.a, sorted_qmm_v.a + qmm_v.n);
  // mm_v.n = mm_h[n];
  // if (para->m <= 5) mm_sketch(km, cons.c_str(), tlen, para->mm_w, para->k, n, 0, &mm_v);
  // mm_h[n + 1] = mm_v.n;


  // sorted_mm_v.n = mm_h[n];
  // for (int i = mm_h[n]; i < (int)mm_v.n; ++i) kv_push(mm128_t, km, sorted_mm_v, mm_v.a[i]);
  // radix_sort_mm128x(sorted_mm_v.a + mm_h[n], sorted_mm_v.a + mm_h[n + 1]);

  // std::cerr << "mm span\n" << mm_h[qid] << "\n";
  // for (int i = mm_h[qid]; i < mm_h[qid + 1]; i++) {
  //   std::cerr << (sorted_mm_v.a[i].x & 0xff) << " ";
  // }
  // std::cerr << "\n";
  // collect anchor between _seq and cons
  // if (para->verbose) std::cerr << "collect anchor between cons and qseq" << "\n";
  mm128_v anchors = { 0, 0, 0 };
  bool use_tr = para->tr_dist > 0;
  if (use_tr) {
    tr_cov(&tmm_v, tlen, para->k, para->tr_dist, tcov, tr_last, tr_iv);
    tr_cov(&qmm_v, qlen, para->k, para->tr_dist, qcov, tr_last, tr_iv);
  }
  collect_anchors(&anchors, qlen, para->max_occ, use_tr ? tcov.data() : nullptr, use_tr ? qcov.data() : nullptr);
  // if (qid == 205) {
  //   std::cerr << "target pos: ";
  //   for (int i = 0; i < anchors.n; i++) {
  //     std::cerr << (int)anchors.a[i].x << " ";
  //   }
  //   std::cerr << "\n";

  //   std::cerr << "query pos: ";
  //   for (int i = 0; i < anchors.n; i++) {
  //     std::cerr << (int)anchors.a[i].y << " ";
  //   }
  //   std::cerr << "\n";
  // }
  if (para->verbose >= 2) std::cerr << "get a optimal chain through dp" << "\n";
  dp_chaining(para, &anchors, tlen, qlen, win_ab);
  // if (para->verbose) {
  //   std::cerr << tlen << "\n";
  //   for (int i = 0; i < anchors.n; i++) {
  //     std::cerr << i << ":" << (int)anchors.a[i].x << " ";
  //   }
  //   std::cerr << "\n";
  // }
  return anchors;
}