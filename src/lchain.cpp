#include "lchain.h"
#include "kalloc.h"
#include "cstring"
#include "kvec.h"
#include <iostream>
#include <algorithm>
/* modified from lh3/minimap2/lchain.c */
/********** start *************/
constexpr int INF = 1e9;
static int64_t mg_chain_bk_end(int32_t max_drop, const mm128_t* z, const int32_t* f, const int64_t* p, int32_t* t, int64_t k)
{
  int64_t i = z[k].y, end_i = -1, max_i = i;
  int32_t max_s = 0;
  if (i < 0 || t[i] != 0) return i;
  do {
    int32_t s;
    t[i] = 2;
    end_i = i = p[i];
    s = i < 0 ? z[k].x : (int32_t)z[k].x - f[i];
    if (s > max_s) max_s = s, max_i = i;
    else if (max_s - s > max_drop) break;
  } while (i >= 0 && t[i] == 0);
  for (i = z[k].y; i >= 0 && i != end_i; i = p[i]) // reset modified t[]
    t[i] = 0;
  return max_i;
}

static inline int32_t comput_sc(const mm128_t* ai, const mm128_t* aj, int32_t max_dist_x, int32_t max_dist_y, int32_t bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg)
{
  int32_t dq = (int32_t)ai->y - (int32_t)aj->y, dr, dd, dg, q_span, sc;
  bool is_equal = false;
  if (dq <= 0 || dq > max_dist_x) {
    return -INF;
  }
  dr = (int32_t)(ai->x - aj->x);
  if (is_equal && (dr == 0 || dq > max_dist_y)) {
    return -INF;
  }
  dd = dr > dq ? dr - dq : dq - dr;
  if (is_equal && dd > bw) {
    return -INF;
  }
  if (n_seg > 1 && !is_cdna && is_equal && dr > max_dist_y) {
    return -INF;
  }
  dg = dr < dq ? dr : dq;
  q_span = ai->y >> 32 & 0xff; // modify aj -> ai
  sc = q_span < dg ? q_span : dg;
  if (dd || dg > q_span) {
    float lin_pen, log_pen;
    lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
    log_pen = dd >= 1 ? mg_log2(dd + 1) : 0.0f; // mg_log2() only works for dd>=2
    if (is_cdna || !is_equal) {
      if (!is_equal && dr == 0) ++sc; // possibly due to overlapping paired ends; give a minor bonus
      else if (dr > dq || !is_equal) sc -= (int)(lin_pen < log_pen ? lin_pen : log_pen); // deletion or jump between paired ends
      else sc -= (int)(lin_pen + .5f * log_pen);
    }
    else sc -= (int)(lin_pen + .5f * log_pen);
  }
  return sc;
}

// can be faster futher if this is be the bottleneck
mm128_t* mg_chain_backtrack(void* km, int64_t n, const int32_t* f, const mm128_t* a, const int64_t* p, int32_t* v, int32_t* t, int32_t min_cnt, int32_t min_sc, int32_t max_drop, int32_t* n_u_, int32_t* n_v_)
{
  mm128_t* z;
  mm128_t* u;
  int64_t i, k, n_z, n_v;
  int32_t n_u;

  *n_u_ = *n_v_ = 0;
  memset(t, 0, n * sizeof(int32_t));

  for (i = n - 1, n_z = 0; i >= 0; i--) {
    if (t[p[i]] == 0 && f[i] >= 0 && f[i] >= min_sc) {
      ++n_z;
      t[i] = 2;
    }
    if (p[i] >= 0) t[p[i]] = 1;
  } // precompute n_z

  if (n_z == 0) return 0;
  z = Kmalloc(km, mm128_t, n_z);

  for (i = 0, k = 0; i < n; ++i) // populate z[]
    if (t[i] == 2) z[k].x = f[i], z[k++].y = i;
  radix_sort_mm128x(z, z + n_z);

  memset(t, 0, n * 4);
  for (k = n_z - 1, n_v = n_u = 0; k >= 0; --k) { // precompute n_u
    if (t[z[k].y] == 0) {
      int64_t n_v0 = n_v, end_i;
      int32_t sc;
      end_i = mg_chain_bk_end(max_drop, z, f, p, t, k);
      for (i = z[k].y; i != end_i; i = p[i])
        ++n_v, t[i] = 1;
      sc = i < 0 ? z[k].x : (int32_t)z[k].x - f[i];
      if (sc >= min_sc && n_v > n_v0 && n_v - n_v0 >= min_cnt)
        ++n_u;
      else n_v = n_v0;
    }
  }

  u = Kmalloc(km, mm128_t, n_u);
  memset(t, 0, n * 4);
  for (k = n_z - 1, n_v = n_u = 0; k >= 0; --k) { // populate u[]
    if (t[z[k].y] == 0) {
      int64_t n_v0 = n_v, end_i;
      int32_t sc;
      end_i = mg_chain_bk_end(max_drop, z, f, p, t, k);
      for (i = z[k].y; i != end_i; i = p[i]) {
        v[n_v++] = i, t[i] = 1;
      }
      sc = i < 0 ? z[k].x : (int32_t)z[k].x - f[i];

      if (sc >= min_sc && n_v > n_v0 && n_v - n_v0 >= min_cnt) {
        uint64_t start_id = v[n_v - 1], end_id = z[k].y;
        uint64_t strand = a[end_id].x >> 63, tpos = (uint32_t)a[end_id].x, qpos = (uint32_t)a[end_id].y;
        u[n_u].x = strand << 63 | tpos << 32 | qpos;
        u[n_u++].y = (uint64_t)n_v0 << 32 | n_v;
        // std::cerr << tpos << " " << qpos << " " << n_v0 << " " << n_v << "\n";
        // for (int i = n_v0; i < n_v; i++) {
        //   int tmp = a[v[i]].x;
        //   std::cerr << tmp << " ";
        // }
        // std::cerr << "\n";
      }
      else n_v = n_v0;
    }
  }

  kfree(km, z);
  // assert(n_v < INT32_MAX);
  *n_u_ = n_u, * n_v_ = n_v;
  return u;
}
static mm128_t* compact_a(void* km, int32_t n_u, mm128_t* u, int32_t n_v, int32_t* v, mm128_t* a)
{
  mm128_t* b, * w;
  int64_t i, j, k;

  // write the result to b[]
  b = Kmalloc(km, mm128_t, n_v);
  for (i = 0, k = 0; i < n_u; ++i) {
    int32_t k0 = k;
    int32_t st = u[i].y >> 32, end = (int32_t)u[i].y;
    // std::cerr << "tpos:" <<( u[i].x >> 32 )<< "\n";
    for (j = end - 1; j >= st; j--) {
      b[k++] = a[v[j]];
      // std::cerr << (int)a[v[j]].x << " ";
    }
    // std::cerr << "\n";
  }
  kfree(km, v);
  kfree(km, a);
  return b;
}
/* Input:
 *   a[].x: rev<<63 | tid<<32 | tpos
 *   a[].y: qid << 48 | flags<<40 | q_span<<32 | q_pos
 * Output:
 * input a[] is deallocated on return
 *   n_u: #chains
 *   u[].x: strand << 63 | tpos << 32 | qpos  a[end_id] 's  tpos and qpos
 *   u[].y: start_id << 32 | end_id id is a 's index,  [start_id, end_id)
 */
mm128_t* mg_lchain_dp(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg, size_t* _n, mm128_t* a, int* n_u_, mm128_t** _u, void* km)
{ // TODO: make sure this works when n has more than 32 bits
  int32_t* f, * t, * v, n_u, n_v, max_drop = bw;
  int64_t* p, i, j, max_ii, st = 0;
  mm128_t* u;
  int64_t n = *_n;
  if (_u) *_u = 0, * n_u_ = 0;
  if (n == 0 || a == 0) {
    kfree(km, a);
    *_n = 0;
    return 0;
  }
  if (max_dist_x < bw) max_dist_x = bw;
  if (max_dist_y < bw && !is_cdna) max_dist_y = bw;
  if (is_cdna) max_drop = INT32_MAX;
  p = Kmalloc(km, int64_t, n);
  f = Kmalloc(km, int32_t, n);
  v = Kmalloc(km, int32_t, n);
  t = Kcalloc(km, int32_t, n);

  // fill the score and backtrack arrays
  for (i = 0, max_ii = -1; i < n; ++i) {
    int64_t max_j = -1, end_j;
    int32_t max_f = a[i].y >> 32 & 0xff, n_skip = 0, n_iter = 0;
    while (st < i && (a[i].x >> 32 != a[st].x >> 32 || a[i].x > a[st].x + max_dist_x)) ++st;
    if (i - st > 5000) st = i - 5000;
    for (j = i - 1; j >= st; --j) {
      int32_t sc;
      sc = comput_sc(&a[i], &a[j], max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg);
      if (sc == -INF) continue;
      sc += f[j];
      if (sc > max_f) {
        max_f = sc, max_j = j;
        if (n_skip > 0) --n_skip;
      }
      else {
        if (sc != max_f) {
          if (++n_iter > max_iter) break;
        }
        if (t[j] == (int32_t)i) {
          if (++n_skip > max_skip) break;
        }
      }
      if (p[j] >= 0) t[p[j]] = i;
    }
    end_j = j;
    // if (max_ii < 0 || a[i].x - a[max_ii].x >(int64_t)max_dist_x) {
    //   int32_t max = -INF;
    //   max_ii = -1;
    //   for (j = i - 1; j >= st; --j)
    //     if (max < f[j]) max = f[j], max_ii = j;
    // }
    // if (max_ii >= 0 && max_ii < end_j) {
    //   int32_t tmp;
    //   tmp = comput_sc(&a[i], &a[max_ii], max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg);
    //   if (tmp != -INF && max_f < tmp + f[max_ii])
    //     max_f = tmp + f[max_ii], max_j = max_ii;
    // }
    f[i] = max_f, p[i] = max_j;
    // if (i == 29) {
    //   std::cerr << "ddl:" << "\n";
    //   std::cerr << (int)a[i].y << " " << (a[i].y >> 32 & 0xff) << "\n";
    //   std::cerr << f[i] << " " << p[i] << "\n";
    // }
    if (max_ii < 0 || (a[i].x - a[max_ii].x <= (int64_t)max_dist_x && f[max_ii] < f[i]))
      max_ii = i;
    // fprintf(stderr, "X1\t%ld\t%ld:%d\t%ld\t%ld:%d\t%ld\t%ld\n", (long)i, (long)(a[i].x>>32), (int32_t)a[i].x, (long)max_j, max_j<0?-1L:(long)(a[max_j].x>>32), max_j<0?-1:(int32_t)a[max_j].x, (long)max_f, (long)v[i]);
  }
  // std::cerr << "mg_lchain_dp score:" << f[max_ii] << "\n";
  u = mg_chain_backtrack(km, n, f, a, p, v, t, min_cnt, min_sc, max_drop, &n_u, &n_v);
  // v restore the anchor 's index
  *n_u_ = n_u, * _u = u; // NB: note that u[] may not be sorted by score here
  kfree(km, p); kfree(km, f); kfree(km, t);
  if (n_u == 0) {
    kfree(km, a); kfree(km, v);
    *_n = 0;
    return 0;
  }
  *_n = n_v;
  return compact_a(km, n_u, u, n_v, v, a);
}
/************ end *************/


inline int get_local_chain_score(int end_tpos_j, int end_qpos_j, int start_anchor_i, int end_anchor_i, mm128_t* a, int* score) {
  int l = start_anchor_i - 1, r = end_anchor_i;
  while (l + 1 < r) {
    int mid = (l + r) / 2;
    int tpos_mid = (uint32_t)a[mid].x, qpos_mid = (uint32_t)a[mid].y;
    if (tpos_mid <= end_tpos_j || qpos_mid <= end_qpos_j) l = mid;
    else r = mid;
  }
  if (r == end_anchor_i) return -INF;
  return score[end_anchor_i - 1] - score[r] + ((a[r].y >> 32) & 0xff); //  + (a[r].y >> 32 & 0xff)
}

/* modified from yangao07/abPOA/abpoa_seed.c */
/********** start *************/
//   a[].x: rev<<63 | tid<<32 | tpos
//   a[].y: qid << 48 | flags << 40 | q_span << 32 | q_pos
// local chains:
//   x: strand | end_tpos | end_qpos
//   y: start_anchor_i | end_anchor_i   [start_anchor_i, end_anchor_i)
int chain_dp(void *km, mm128_t *lchains, int n_lchains, mm128_v *_anchors, int max_dist_x, int max_dist_y, int bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg) {
  size_t n = _anchors->n;

  mm128_t *a = _anchors->a;
  int32_t *f = Kmalloc(km, int32_t, n);
  int *chain_score = Kmalloc(km, int, n_lchains), *pre_chain = Kmalloc(km, int, n_lchains);
  int global_max_score = -INF, global_max_i = -1;
  for (int i = 0, st = 0; i < n_lchains; i++) {
    uint64_t xi = lchains[i].x, yi = lchains[i].y;
    int strand = xi >> 63, end_qpos_i = (int32_t)xi, start_anchor_i = yi >> 32, end_anchor_i = (int32_t)yi;
    int start_tpos_i = (int32_t)a[start_anchor_i].x, start_qpos_i = (int32_t)a[start_anchor_i].y;
    f[start_anchor_i] = a[start_anchor_i].y >> 32 & 0xff;
    for (int j = start_anchor_i + 1; j < end_anchor_i; j++) {
      int32_t sc;
      sc = comput_sc(&a[j], &a[j - 1], max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg);
      f[j] = f[j - 1] + sc;
    }
    int max_j = -1, max_score = f[end_anchor_i - 1];
    while (st < i) {
      if ((int)((lchains[st].x) >> 63) != strand) ++st;
      else break;
    }
    for (int j = i - 1; j >= st; --j) {
      uint64_t xj = lchains[j].x, yj = lchains[j].y;
      int end_tpos_j = (xj >> 32) & 0x7fffffff, end_qpos_j = (int32_t)xj; //, j_end_anchor_i = iy >> 32;
      int start_anchor_j = yj >> 32;
      int start_tpos_j = (int32_t)a[start_anchor_j].x, start_qpos_j = (int32_t)a[start_anchor_j].y;
      if (start_tpos_i < start_tpos_j) continue;
      int score1 = -INF;
      if (start_tpos_i > end_tpos_j && start_qpos_i > end_qpos_j) score1 = chain_score[j] + f[end_anchor_i - 1];
      else score1 = chain_score[j] + get_local_chain_score(end_tpos_j, end_qpos_j, start_anchor_i, end_anchor_i, a, f);
      // std::cerr << "tpos:" << start_tpos_i << " " << start_tpos_j << "\n";
      // std::cerr << "detail:" << i << " " << j << " " << score1 << "\n";
      if (score1 > max_score) {
        max_score = score1; max_j = j;
      }
    }
    chain_score[i] = max_score; pre_chain[i] = max_j;
    // std::cerr << i << " " << f[end_anchor_i - 1] << "\n";
    // std::cerr << i << " " << max_score << " " << max_j << "\n";

    if (max_score > global_max_score) {
      global_max_score = max_score;
      global_max_i = i;
    }
  }
  if (global_max_i < 0) {
    kfree(km, f);
    kfree(km, chain_score);
    kfree(km, pre_chain);
    return 0;
  }
  mm128_v anchors = { 0, 0, 0 };
  // collect anchors based on global_max_i
  int cur_i = global_max_i, pre_i = pre_chain[global_max_i];
  uint64_t cur_y = lchains[cur_i].y, pre_x, pre_y;
  int i = (uint32_t)cur_y - 1;
  while (pre_i != -1) { // collect valid anchors in local_chains[cur_i], constrained by local_chains[pre_i]
    pre_x = lchains[pre_i].x, pre_y = lchains[pre_i].y;

    int pre_end_tpos = (pre_x >> 32) & 0x7fffffff, pre_end_qpos = (int32_t)pre_x;

    while (i >= 0) {
      int cur_tpos = a[i].x, cur_qpos = (int32_t)a[i].y;

      if (cur_tpos > pre_end_tpos && cur_qpos > pre_end_qpos) {
        kv_push(mm128_t, km, anchors, a[i]);
      }
      else break;
      i--;
      if (i == (cur_y >> 32)) break;
    }
    cur_i = pre_i, pre_i = pre_chain[pre_i], cur_y = pre_y;
    i = (uint32_t)cur_y - 1;
  }
  // collect anchors of last chain: local_chains[cur_i]
  // pre_end_tpos = (pre_x >> 32) & 0x7fffffff, pre_end_qpos = (int32_t)pre_x;
  while (i >= 0) {
    kv_push(mm128_t, km, anchors, a[i]);
    if (i == (cur_y >> 32)) break;
    i--;
  }
  // reverse order of par_anchors
  for (i = 0; i < (int)(anchors.n) >> 1; ++i) {
    mm128_t tmp = anchors.a[i];
    anchors.a[i] = anchors.a[anchors.n - i - 1];
    anchors.a[anchors.n - i - 1] = tmp;
  }

  // if (verbose >= ABPOA_LONG_DEBUG_VERBOSE) {
  //   for (i = _n; i < par_anchors->n; ++i) {
  //     uint64_t ia = par_anchors->a[i];
  //     // strand, rpos, qpos
  //     fprintf(stderr, "%c\t%" PRIu64 "\t%d\n", "+-"[ia >> 63], (ia >> 32) & 0x7fffffff, ((uint32_t)ia));
  //   }
  // }
  kfree(km, _anchors->a);
  *_anchors = anchors;
  kfree(km, f);
  kfree(km, chain_score), kfree(km, pre_chain);
  return 0;
}
/************ end *************/

// 按累计漂移划分 POA 窗口（参照 k-band 分段）
//   锚点对角线 d = t - q；以窗口起点锚点 s 为基准，累计漂移 (t_i - t_s) - (q_i - q_s) = d_i - d_s，
//   带符号累计，插入与缺失可相互抵消。
//   从 s 往后逐个检查 |d_i - d_s| <= max_drift、与前一锚点的距离 <= max_drift、窗口长度 max(gt, gq) <= max_win，
//   首次越界即停止，取最远的不与 s 重叠的合格锚点作为窗口终点，该窗口用静态带；
//   相邻锚点之间可能藏着相互抵消的插入与缺失，锚点处看不到其偏移，最大可达两锚点的距离，
//   因此距离超过 max_drift 与漂移越界同样处理。
//   若第一个候选就越界，则以下一个不与 s 重叠的锚点强制成窗：漂移越界或距离超过 max_drift 用自适应带。
// 输入：链上全部锚点（t 升序）；输出：anchors 原地压缩为窗口边界锚点，
//   win_ab[i] 表示以第 i 个边界锚点为终点的窗口是否用自适应带，win_ab[n] 为尾窗口。
// 索引 -1 为虚拟起点 V0（te = qe = -1），n 为虚拟终点 Vend（ts = tlen, qs = qlen）。
void partition_windows(mm128_v *anchors, int tlen, int qlen, int max_drift, int max_win, std::vector<bool> *win_ab) {
  mm128_t *a = anchors->a;
  int n = anchors->n;
  auto te = [&](int i) -> int64_t { return i < 0 ? -1 : (int32_t)a[i].x; };
  auto qe = [&](int i) -> int64_t { return i < 0 ? -1 : (int32_t)a[i].y; };
  auto ts = [&](int i) -> int64_t { return i == n ? tlen : te(i) - (int)(a[i].y >> 32 & 0xff) + 1; };
  auto qs = [&](int i) -> int64_t { return i == n ? qlen : qe(i) - (int)(a[i].y >> 32 & 0xff) + 1; };
  auto diag = [&](int i) -> int64_t { return i < 0 ? 0 : i == n ? (int64_t)tlen - qlen : te(i) - qe(i); };

  win_ab->clear();
  int m = 0;  // 已选出的边界锚点数
  for (int s = -1; s < n;) {
    int64_t ds = diag(s);
    int e = -2;  // -2 表示没有合格锚点
    for (int i = s + 1; i <= n; i++) {
      int64_t gt = ts(i) - te(s) - 1, gq = qs(i) - qe(s) - 1;
      int64_t drift = diag(i) - ds;
      int64_t gap = std::max(ts(i) - te(i - 1) - 1, qs(i) - qe(i - 1) - 1);  // 与前一锚点的距离（参考/query 两侧取大）
      if (drift > max_drift || drift < -max_drift || gap > max_drift || std::max(gt, gq) > max_win) break;
      if (gt >= 0 && gq >= 0) e = i;
    }
    bool ab = false;
    if (e == -2) {  // 强制单步窗口：跳到下一个不重叠锚点（Vend 必然满足）
      for (e = s + 1; e < n; e++) {
        if (ts(e) - te(s) - 1 >= 0 && qs(e) - qe(s) - 1 >= 0) break;
      }
      int64_t drift = diag(e) - ds;
      int64_t gap = std::max(ts(e) - te(s) - 1, qs(e) - qe(s) - 1);
      ab = drift > max_drift || drift < -max_drift || gap > max_drift;
    }
    win_ab->push_back(ab);
    if (e < n) a[m++] = a[e];
    s = e;
  }
  anchors->n = m;
}
