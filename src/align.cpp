#include "align.h"
#include "simdpriv.h"
#include <cassert>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <omp.h>
// #include "timer.h"

constexpr int INF = 0x3f3f3f3f; // 0x3f3f3f3f
constexpr int NEG_INF = 0xc0c0c0c0; // 0xc0c0c0c0
constexpr int M_OP = 1, D_OP = 2, I_OP = 4, ALL_OP = 7;

inline int calj(int acj, int Bs) {
  return  acj - Bs * simd_width;
}

std::vector<res_t> poa(const para_t *para, const graph *DAG, int beg_id, int end_id, int qid, const char *qseq, int qlen, aligned_buff_t *mpool, bool ab_band) {

  // Timer::instance().start("init");
  std::chrono::microseconds total_part1(0);
  auto start1 = std::chrono::high_resolution_clock::now();

  const std::vector<node_t> &node = DAG->node;const std::vector<int> &rank = DAG->rank;
  int beg_i = node[beg_id].rank, end_i = node[end_id].rank;
  int n = end_i - beg_i + 1, m = qlen + 2;
  ab_band |= para->ab_band;

  std::string seq;
  seq += char26_table['N'];
  for (int j = 0; j < qlen; j++) {
    seq += char26_table[(int)qseq[j]];
  }
  seq += char26_table['N'];
  int col_size = (m + 1 + simd_width - 1) / simd_width * simd_width;
  seq.append(col_size - m, char26_table['N']);
  std::vector<res_t> res;

  int para_m = para->m, e1 = para->gap_ext1, o1 = para->gap_open1 + e1;
  simd_reg E1 = simd_set1(e1), O1 = simd_set1(o1), Neg_inf = simd_set1(NEG_INF);
  // std::cout << mat << " " << mis << " " << o1 << " " << e1 << "\n";

  std::vector<int> Ms(n, m + 1), Me(n); // index is rank  [Ms[i], Me[i]]
  std::vector<int> Bs(n), Be(n);  // index is rank    [Bs[i], Be[i]) 
  std::vector<int> Mp(n), Sl(n), Pl(n, m), Pr(n), Ol(n), Or(n), Ow(n);  // index is rank
  size_t mtx_size = 0;
  int w = para->b;
  if (para->f) w += m / para->f;

  std::vector<int> hlen(n, NEG_INF), tlen(n, NEG_INF);
  hlen[0] = 0;
  std::vector<int> score(n);  // index is rank
  for (int i = 1; i < n; i++) { // ni topsort id dp
    int u = rank[i + beg_i];
    const node_t &cur = node[u];
    int wmax = -1, max_pre = -1;
    for (size_t k = 0; k < cur.in.size(); k++) {
      int v = cur.in[k];
      int pre = node[v].rank - beg_i;
      if (pre < 0 || hlen[pre] == NEG_INF) continue;
      if (cur.in_weight[k] > wmax || (cur.in_weight[k] == wmax && score[pre] > score[max_pre])) {
        wmax = cur.in_weight[k];
        max_pre = pre;
      }
    }
    if (max_pre != -1) score[i] = score[max_pre] + wmax;
    if (max_pre != -1) hlen[i] = hlen[max_pre] + 1;
    // std::cerr << u << " " << lp[i] << " " << rp[i] << "\n";
  }
  score.resize(node.size());
  // tlen[node[1].rank] = 1;
  tlen[n - 1] = 1;
  for (int i = n - 2; i >= 0; i--) { // ni topsort id dp
    int u = rank[i + beg_i];
    const node_t &cur = node[u];
    int wmax = -1, max_suc = -1;
    for (size_t k = 0; k < cur.out.size(); k++) {
      int v = cur.out[k];
      int suc = node[v].rank - beg_i;
      if (suc >= n || tlen[suc] == NEG_INF) continue;
      if (cur.out_weight[k] > wmax || (cur.out_weight[k] == wmax && score[suc] > score[max_suc])) {
        wmax = cur.out_weight[k];
        max_suc = suc;
      }
    }
    if (max_suc != -1) score[i] = score[max_suc] + wmax;
    if (max_suc != -1) tlen[i] = tlen[max_suc] + 1;

    // std::cerr << u << " " << lp[i] << " " << rp[i] << "\n";
  }
  // lp[node[0].rank] = 0; // src lp = 0

  // para->m * m个int? para->mat
  size_t p_size = para_m * col_size;
  size_t invisible_size = 0;
  for (int i = 0; i < n; i++) {
    // Ms[i] = 0, Me[i] = m;
    if (para->f > 0 && !ab_band) {
      Ms[i] = std::max(0, std::min(hlen[i], m - tlen[i]) - w), Me[i] = std::min(m, std::max(hlen[i], m - tlen[i]) + w);
    }
    else {
      Ms[i] = 0, Me[i] = m;
    }
    Bs[i] = Ms[i] / simd_width, Be[i] = Me[i] / simd_width + 1; // [block_s,block_e)
    int offset = (Be[i] - Bs[i]) * simd_width;
    if (hlen[i] == NEG_INF || tlen[i] == NEG_INF) offset = 0;
    else invisible_size += 2 * simd_width;
    mtx_size += offset;
    // sum += Me[i] - Ms[i] + 1;
  }
  size_t sum = 0;
  void *buff = nullptr;
  if (para->verbose >= 2) std::cerr << "mtx size:" << (p_size + 3 * mtx_size) * sizeof(int) / 1024 / 1024 / 1024 << "GB" << "\n";
  if (mpool != nullptr) mpool->alloc_aligned(&buff, SIMD_BYTES, (p_size + 3 * mtx_size + invisible_size) * sizeof(int));
  else alloc_aligned(&buff, SIMD_BYTES, (p_size + 3 * mtx_size + invisible_size) * sizeof(int)); // malloc

  std::vector<int *> P(para_m);
  for (int i = 0; i < para_m; i++) {
    P[i] = (int *)buff + i * col_size;
    for (int j = 0; j < col_size; j++) {
      P[i][j] = para->mat[i * para_m + seq[j]];
    }
  }
  // std::cerr << "ddl" << "\n";
  // std::vector<int*> M(n), D(n), I(n);
  // M[0] = (int*)buff + p_size;
  // D[0] = M[0] + mtx_size + invisible_size;
  // I[0] = D[0] + mtx_size;

  // for (int i = 1; i < n; i++) {
  //   int offset = (Be[i - 1] - Bs[i - 1]) * simd_width;
  //   int invisible_offset = 0;
  //   if (hlen[i - 1] == NEG_INF || tlen[i - 1] == NEG_INF) offset = 0;
  //   else invisible_offset = simd_width;
  //   M[i] = M[i - 1] + offset + invisible_offset;
  //   D[i] = D[i - 1] + offset;
  //   I[i] = I[i - 1] + offset;
  // }
  std::vector<int *> M(n), D(n), I(n); // 同一个 i 的 M, D, I 紧挨着排列
  int *current_ptr = (int *)buff + p_size; // 跳过 Profile(P) 矩阵占用的空间
  for (int i = 0; i < n; i++) {
    int offset = (Be[i] - Bs[i]) * simd_width;
    int invisible_offset = 0;
    if (hlen[i] == NEG_INF || tlen[i] == NEG_INF) offset = 0;
    else invisible_offset = 2 * simd_width;
    M[i] = current_ptr + simd_width;
    // M 的末尾保留 invisible_offset，防止后续 simd_loadu(M_p + pj - 1) 越界读取
    D[i] = M[i] + offset + simd_width;
    I[i] = D[i] + offset;
    current_ptr += (offset * 3) + invisible_offset;
  }

  if (para->f > 0 && ab_band) {
    // Ms[i] = std::max(0, std::min(DAG->hlen[aci] - DAG->hlen[beg_i], m + DAG->tlen[aci] - DAG->tlen[end_i]) - w), Me[i] = std::min(m, std::max(DAG->hlen[aci] - DAG->hlen[beg_i], m + DAG->tlen[aci] - DAG->tlen[end_i]) + w);
    Ms[0] = std::max(0, std::min({ Pl[0], hlen[0] + Ol[0], m - tlen[0] }) - w), Me[0] = std::min(m, std::max({ Pr[0], hlen[0] + Or[0], m - tlen[0] }) + w);
    Bs[0] = Ms[0] / simd_width, Be[0] = Me[0] / simd_width + 1; // [block_s,block_e)
  }

  // dp init
  int block_num = Be[0] - Bs[0]; // not contain Be[i] - 1 's block1
  int *M_i = M[0];
  int *D_i = D[0];
  int *I_i = I[0];
  for (int bid = 0; bid < block_num; bid++) {
    // memset(M, 0xc0, col_size * sizeof(int));//M[0] = NEG_INF
    // memset(D, 0xc0, col_size * sizeof(int));//D[0] = NEG_INF
    simd_store(M_i + bid * simd_width, Neg_inf);
    simd_store(D_i + bid * simd_width, Neg_inf);
    // simd_store(I_i + bid * simd_width, Neg_inf);
  }
  simd_store(M_i - simd_width, Neg_inf);
  simd_store(M_i + block_num * simd_width, Neg_inf);
  M_i[0] = P[char26_table['N']][0];
  I_i[0] = NEG_INF;
  for (int j = 1; j < block_num * simd_width; j++) { // block_num * reg_size
    I_i[j] = std::max(I_i[j - 1] + e1, M_i[j - 1] + o1); // isource
    M_i[j] = std::max({ M_i[j], D_i[j], I_i[j] });  // three source
  }
  // simd_store(I[0] + block_num * simd_width, Neg_inf);

  // int max = 0;
  // int offset_band = 0;
  // Timer::instance().stop("init");
  for (int i = 1; i < n; i++) {
    // std::cerr << i << " " << n << "\n";
    int aci = beg_i + i;
    const node_t &cur = node[rank[aci]];
    // std::cerr << aci << " " << cur.rank << " " << char256_table[cur.base] << "\n";
    if (hlen[i] == NEG_INF || tlen[i] == NEG_INF) continue;
    M_i = M[i];
    D_i = D[i];
    I_i = I[i];
    if (para->f > 0 && ab_band) {  // adptive band
      // int pmid = (Pl[i] + Pr[i]) / 2;
      // int tms = std::min({ Pl[i], DAG->hlen[i] + Ol[i], m - DAG->tlen[i] }), tme = std::max({ Pr[i], DAG->hlen[i] + Or[i], m - DAG->tlen[i] });
      // int len = std::max(pmid - tms + 1, tme - pmid);
      // tms = std::max(0, pmid - len - w - Ow[i]), tme = std::min(m, pmid + len + w + Ow[i]);
      // Ms[i] = tms, Me[i] = tme;
      // int tw = para->b + (m - (DAG->hlen[i] + Ol[i])) / para->f;

      // Ms[i] = std::max(0, std::min({ Pl[i], DAG->hlen[i] + Ol[i], m - DAG->tlen[i] }) - w - Ow[i]), Me[i] = std::min(m, std::max({ Pr[i], DAG->hlen[i] + Or[i], m - DAG->tlen[i] }) + w + Ow[i]);
      Ms[i] = std::max(0, std::min({ Pl[i], hlen[i] + Ol[i], m - tlen[i] }) - w - Ow[i]), Me[i] = std::min(m, std::max({ Pr[i], hlen[i] + Or[i], m - tlen[i] }) + w + Ow[i]);
      Bs[i] = Ms[i] / simd_width, Be[i] = Me[i] / simd_width + 1; // [block_s,block_e)
    }
    // std::cerr << Bs[i] << " " << Be[i] << "\n";
    int block_num = Be[i] - Bs[i]; // not contain Be[i] - 1 's block
    sum += Me[i] - Ms[i] + 1;
    int pre_num = 0;
    // Timer::instance().start("MD");
    for (size_t k = 0; k < cur.in.size(); k++) {
      const node_t &pre = node[cur.in[k]];
      int p = pre.rank - beg_i; // rank
      if (p < 0 || hlen[p] == NEG_INF) continue;

      int *M_p = M[p];
      int *D_p = D[p];
      for (int bid = 0; bid < block_num; bid++) { // SIMD
        int j = bid * simd_width;
        int acBid = (Bs[i] + bid);
        int acj = acBid * simd_width;
        simd_reg Mij = pre_num == 0 ? Neg_inf : simd_load(M_i + j);
        simd_reg Dij = pre_num == 0 ? Neg_inf : simd_load(D_i + j);
        int pj = calj(acj, Bs[p]);

        if (pj >= 0 && acBid < Be[p]) {
          simd_reg TMP = simd_loadu(M_p + pj - 1);
          Mij = simd_max(Mij, TMP);

          TMP = simd_load(D_p + pj);
          Dij = simd_max(Dij, simd_add(TMP, E1));

          TMP = simd_load(M_p + pj);
          Dij = simd_max(Dij, simd_add(TMP, O1));
        }
        simd_store(M_i + j, Mij);
        simd_store(D_i + j, Dij);
      }
      pre_num++;
    }
    // Timer::instance().stop("MD");

    if (pre_num == 0) {
      for (int bid = 0; bid < block_num; bid++) {
        simd_store(M_i + bid * simd_width, Neg_inf);
        simd_store(D_i + bid * simd_width, Neg_inf);
      }
    }
    else {
      // Timer::instance().start("Profile_I_Fusion");
      int cur_base = i != n - 1 ? cur.base : char26_table['N'];
      I_i[0] = NEG_INF;

      for (int bid = 0; bid < block_num; bid++) {
        int j = bid * simd_width;
        int acBid = (Bs[i] + bid);
        int acj = acBid * simd_width;

        // 1. 批量加载刚才 MD 循环算好的 M_i 和 D_i
        simd_reg Mij = simd_load(M_i + j);
        simd_reg Dij = simd_load(D_i + j);

        Mij = simd_add(Mij, simd_load(P[cur_base] + acj));

        alignas(SIMD_BYTES) int m_arr[simd_width];
        alignas(SIMD_BYTES) int d_arr[simd_width];
        simd_store(m_arr, Mij);
        simd_store(d_arr, Dij);

        // 4. 无痛执行纯标量 I 循环 (完全在栈上跑，避开堆内存的 Store-Forwarding Stall)
        for (int v = 0; v < simd_width; v++) {
          int tj = j + v;
          // if (tj == block_num * simd_width - 1) break; // 无需break, 因为I会用到 下一个 M的左padding,不会越界
          int md = std::max(m_arr[v], d_arr[v]);
          I_i[tj + 1] = std::max(I_i[tj] + e1, md + o1);
          m_arr[v] = std::max(md, I_i[tj]);
        }
        simd_store(M_i + j, simd_load(m_arr));
      }
      // Timer::instance().stop("Profile_I_Fusion");
    }

    // --- 边界复原 ---

    simd_store(M_i - simd_width, Neg_inf);
    simd_store(M_i + block_num * simd_width, Neg_inf);
    // simd_store(D_i + block_num * simd_width, Neg_inf);
    // Timer::instance().stop("MD");
    // Timer::instance().start("I");
    // I_i[0] = NEG_INF;
    // // M_i[0] = std::max({ M_i[0], D_i[0], I_i[0] });
    // // for (int j = 1; j < block_num * simd_width; j++) { // block_num * reg_size
    // //   I_i[j] = std::max(I_i[j - 1] + e1, M_i[j - 1] + o1); // isource
    // //   M_i[j] = std::max({ M_i[j], D_i[j], I_i[j] });  // three source
    // // }
    // for (int j = 0; j < block_num * simd_width - 1; j++) { // block_num * reg_size
    //   int md = std::max(M_i[j], D_i[j]);
    //   I_i[j + 1] = std::max(I_i[j] + e1, md + o1); // isource
    //   M_i[j] = std::max(md, I_i[j]);  // three source
    // }
    // // simd_store(I_i + block_num * simd_width, Neg_inf);
    // Timer::instance().stop("I");
    if (para->f > 0 && ab_band) {
      // Timer::instance().start("argmax");
      int max_j = simd_argmax(M_i, block_num);
      // Timer::instance().stop("argmax");
      // for (int j = 1; j < block_num *simd_width; j++) { // block_num * reg_size
      //   max_j = M_i[j] > M_i[max_j] ? j : max_j;
      // }
      int max_acj = Bs[i] * simd_width + max_j, max_slp = 0;
      bool isMatch = false;
      Mp[i] = max_acj;
      if (cur.base == seq[max_acj]) {
        for (size_t k = 0; k < cur.in.size(); k++) {
          int p = node[cur.in[k]].rank - beg_i; // rank
          if (p < 0 || hlen[p] == NEG_INF) continue;
          max_slp = Sl[p] > max_slp ? Sl[p] : max_slp;
          if (Mp[p] + 1 == Mp[i]) {
            Sl[i] = std::max(Sl[i], Sl[p] + 1);
            isMatch = true;
          }
        }
      }
      if (!isMatch && max_slp < 30) {
        // w += max_slp;
        Ow[i] += max_slp + 1;
        Ow[i] = std::min(Ow[i], 5 * w);
      }
      if (Sl[i] >= 30) {
        // offset_band = max_acj - DAG->hlen[i];
        Ol[i] = max_acj - hlen[i];
        Or[i] = max_acj - hlen[i];
        Ow[i] = 0;
        // Sl[i] = 0;
        for (size_t k = 0; k < cur.out.size(); k++) {
          int suc = node[cur.out[k]].rank - beg_i;
          if (suc >= n) continue;
          Ol[suc] = Ol[i], Or[suc] = Or[i];
        }
      }
      for (size_t k = 0; k < cur.out.size(); k++) {
        int suc = node[cur.out[k]].rank - beg_i;
        if (suc >= n) continue;
        Ol[suc] = std::min(Ol[suc], Ol[i]), Or[suc] = std::max(Or[suc], Or[i]);
      }

      for (size_t k = 0; k < cur.out.size(); k++) {
        int suc = node[cur.out[k]].rank - beg_i;
        if (suc >= n) continue;
        Pl[suc] = std::min(Pl[suc], max_acj), Pr[suc] = std::max(Pr[suc], max_acj + 1);
        Ow[suc] = Ow[i] ? std::max(Ow[suc], Ow[i]) : Ow[i];
        // int tms = std::min({ Pl[suc], DAG->hlen[suc] + Ol[suc], m - DAG->tlen[suc] }), tme = std::max({ Pr[suc], DAG->hlen[suc] + Or[suc], m - DAG->tlen[suc] });
        // int tbs = tms / reg_size, tbe = tme / reg_size;
        // Pr[suc] = std::max(Pr[suc], max_acj + (max_acj - tbs * reg_size));
        // Pl[suc] = std::min(Pl[suc], max_acj - (tbe * reg_size - max_acj));
      }
    }
  }
  // Timer::instance().start("traceback");

  auto end1 = std::chrono::high_resolution_clock::now();

  int i = n - 1, acj = m - 1;
  int j = calj(acj, Bs[i]);
  // int ans = M[i][j];
  int op = ALL_OP;
  if (para->verbose >= 2) std::cerr << "band mode:" << ab_band << "\n";
  if (para->verbose >= 2) std::cerr << "score:" << M[i][j] << "\n";
  total_part1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  if (para->verbose >= 2) std::cerr << "used time: " << total_part1.count() / 1000 << " ms\n";
  while (i > 0 || acj > 0) {
    // dsource
    j = calj(acj, Bs[i]);
    int aci = beg_i + i;
    if (acj >= Be[i] * simd_width || acj < Bs[i] * simd_width) {
      std::cerr << "l:" << Bs[i] * simd_width << " " << "r:" << Be[i] * simd_width << "\n";
      std::cerr << acj << "\n";
      exit(0);
    }
    const node_t &cur = node[rank[aci]];
    int cur_base = i != n - 1 ? cur.base : char26_table['N'];
    int p_score = para->mat[cur_base * para_m + seq[acj]];
    if (op & M_OP && acj > 0) {  // M
      int bk = -1;
      if (cur_base == seq[acj]) {
        for (size_t k = 0; k < cur.in.size(); k++) {
          const node_t &pre = node[cur.in[k]];
          int p = pre.rank - beg_i; // rank
          if (p < 0 || hlen[p] == NEG_INF) continue;
          // M
          int pj = calj(acj, Bs[p]);
          int block_num = Be[p] - Bs[p]; // not contain Be[i] - 1 's block
          if (pj - 1 >= 0 && pj - 1 < block_num * simd_width && M[i][j] == M[p][pj - 1] + p_score && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
            bk = k;
            // break;
          }
        }
      }
      if (bk != -1 && cur.in_weight[bk] >= cur.ind / 10) {  // backtrack based on the normal sample
        const node_t &pre = node[cur.in[bk]];
        int p = pre.rank - beg_i; // rank
        if (i == n - 1 && acj != m - 1) {
          res.emplace_back(res_t(-1, seq[acj]));
        }
        else if (cur_base == seq[acj]) {
          // std::cout << "M";
          res.emplace_back(res_t(cur.id, cur.base));
        }
        else {
          // std::cout << "X";
          const node_t &par = node[cur.par_id]; // dsu.find par
          if (par.aligned_node[seq[acj]] != -1) {
            res.emplace_back(res_t(par.aligned_node[seq[acj]], seq[acj]));
          }
          else res.emplace_back(res_t(-1, seq[acj], cur.par_id));
        }
        op = ALL_OP;
        i = p, acj--;
        continue;
      }
    }
    if (op & D_OP) {
      if (op == D_OP || M[i][j] == D[i][j]) {
        // std::cerr << M[i][j] << " " << D[i][j] << "\n";
        const node_t &cur = node[rank[aci]];
        // std::cerr << M[i * col_size + j] << " " << D[i * col_size + j] << "\n";
        int bk = -1;char bop;
        for (size_t k = 0; k < cur.in.size(); k++) {
          const node_t &pre = node[cur.in[k]];
          int p = pre.rank - beg_i; // rank
          if (p < 0 || hlen[p] == NEG_INF) continue;
          int pj = calj(acj, Bs[p]);
          if (pj >= 0 && acj / simd_width < Be[p]) {
            // std::cerr << D[i][j] << " " << M[p][pj] << " " << o1 << "\n";
            if (D[i][j] == M[p][pj] + o1 && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
              bk = k;
              bop = M_OP | I_OP;
              // break;
            }
          }
        }
        if (bk == -1) {
          for (size_t k = 0; k < cur.in.size(); k++) {
            const node_t &pre = node[cur.in[k]];
            int p = pre.rank - beg_i; // rank
            if (p < 0 || hlen[p] == NEG_INF) continue;
            int pj = calj(acj, Bs[p]);
            if (pj >= 0 && acj / simd_width < Be[p]) {
              if (D[i][j] == D[p][pj] + e1 && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
                bk = k;
                bop = D_OP;
                // break;
              }
            }
          }
        }
        // std::cerr << bk << "\n";
        if (bk != -1) {
          i = node[cur.in[bk]].rank - beg_i;
          op = bop;
          continue;
        }
      }
    }
    if (op & I_OP && acj > 0) {
      if (j - 1 >= 0) {
        if (op == I_OP || M[i][j] == I[i][j]) {
          // std::cerr << "I";
          if (I[i][j] == M[i][j - 1] + o1) {
            res.emplace_back(res_t(-1, seq[acj]));
            op = M_OP | D_OP;
            acj--;
            continue;
          }
          if (I[i][j] == I[i][j - 1] + e1) {
            res.emplace_back(res_t(-1, seq[acj]));
            op = I_OP;
            acj--;
            continue;
          }
        }
      }
    }
    if (op & M_OP && acj > 0) {  // MX
      const node_t &cur = node[rank[aci]];
      int bk = -1;
      for (size_t k = 0; k < cur.in.size(); k++) {
        const node_t &pre = node[cur.in[k]];
        int p = pre.rank - beg_i; // rank
        // if (ans == -2314 && M[i][j] == -18) {
        //   std::cerr << i << " " << j << "\n";
        //   std::cerr << k << " " << p << "\n";
        // }
        if (p < 0 || hlen[p] == NEG_INF) continue;
        // if (pre.base != seq[acj - 1]) continue;
        // M
        int pj = calj(acj, Bs[p]);
        // if (ans == -2314 && M[i][j] == -18  && pj - 1 >= 0)
        //   std::cerr << M[i][j] << " " << M[p][pj - 1] << " " << p_score << "\n";
        int block_num = Be[p] - Bs[p]; // not contain Be[i] - 1 's block
        if (pj - 1 >= 0 && pj - 1 < block_num * simd_width && M[i][j] == M[p][pj - 1] + p_score && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
          bk = k;
          // break;
        }
      }
      if (bk != -1) {
        // std::cerr << "M";
        const node_t &pre = node[cur.in[bk]];
        int p = pre.rank - beg_i; // rank
        if (i == n - 1 && acj != m - 1) {
          res.emplace_back(res_t(-1, seq[acj]));
        }
        else if (cur_base == seq[acj]) {
          // std::cout << "M";
          res.emplace_back(res_t(cur.id, cur.base));
        }
        else {
          // std::cout << "X";
          const node_t &par = node[cur.par_id]; // dsu.find par
          if (par.aligned_node[seq[acj]] != -1) {
            res.emplace_back(res_t(par.aligned_node[seq[acj]], seq[acj]));
          }
          else res.emplace_back(res_t(-1, seq[acj], cur.par_id));
        }
        op = ALL_OP;
        i = p, acj--;
        continue;
      }
    }
    // std::cerr << n << " "<< m << " " << beg_id << " " << end_id << "\n";
    // std::cerr << aci << " " << acj << "\n";
    std::cerr << " backtrack error" << "\n";
    // std::cerr << ans << " " << i << " " << j << "\n";
    // std::cerr << M[i][j] << " " << D[i][j] << " " << I[i][j] << " "<< op << "\n";
    exit(1);

  }
  res.emplace_back(res_t(beg_id, node[beg_id].base));
  if (mpool == nullptr) free_aligned(buff);
  std::reverse(res.begin(), res.end()); //[,)
  res.pop_back();
  // Timer::instance().stop("traceback");
  return res;
}
// #define sort_key_mm128x(a) ((a).x)
// KRADIX_SORT_INIT(mm128x, mm128_t, sort_key_mm128x, 8)

std::vector<res_t> alignment(const para_t *para, graph *DAG, minimizer_t *mm, int rid, const char *qseq, int qlen, aligned_buff_t *mpool) {
  std::string tseq;
  for (int j = 0; j < qlen; j++) {
    tseq += char26_table[(int)qseq[j]];
  }
  std::vector<res_t> res;
  if (DAG->node.size() <= 2) {
    res.emplace_back(res_t(0, char26_table['N']));
    for (size_t j = 0; j < tseq.size(); j++) {
      res.emplace_back(res_t(-1, tseq[j]));
    }
    return res;
  }
  if (!para->enable_seeding) {
    return  poa(para, DAG, 0, 1, rid, tseq.c_str(), tseq.size(), mpool, para->ab_band);
    // return abPOA(para, DAG, mm, rid, tseq, mpool);
  }
  // para->enable_seeding
  if (!DAG->is_topsorted) DAG->topsort(para, 0);

  if (para->verbose >= 2) std::cerr << "collect_anchors_bycons" << "\n";
  mm128_v anchors = mm->collect_anchors_bytseq(para, rid, qseq, qlen, mm->seqs_size, DAG->cons.c_str(), DAG->cons.size()); // tseq is cons
  // no chain
  bool ab_band = 1;
  if (anchors.n <= 0) {
    return poa(para, DAG, 0, 1, rid, tseq.c_str(), tseq.size(), mpool, ab_band);
    // return abPOA(para, DAG, mm, rid, tseq, mpool);
  }
  if (para->thread <= 1) {
    int beg_id = 0, beg_qpos = 0, end_id = -1, end_tpos = -1, end_qpos = -1;
    int j = 0;
    if (para->verbose >= 2) std::cerr << "subgraph poa" << "\n";
    for (size_t i = 0; i < anchors.n; i++) {
      uint64_t xi = anchors.a[i].x, yi = anchors.a[i].y;
      int q_span = yi >> 32 & 0xff;
      end_tpos = xi - q_span + 1, end_qpos = yi - q_span + 1;
      end_id = DAG->cons_pos_to_id[end_tpos];
      int qlen = end_qpos - beg_qpos;
      std::vector<res_t> t_res = poa(para, DAG, beg_id, end_id, rid, tseq.c_str() + j, qlen, mpool, ab_band);
      res.insert(res.end(), t_res.begin(), t_res.end());
      j += qlen;
      for (int k = 0; k < q_span; k++, j++) {
        end_id = DAG->cons_pos_to_id[end_tpos + k];
        if (k != q_span - 1) {
          const node_t &cur = DAG->node[end_id];
          res.emplace_back(res_t(cur.id, cur.base));
        }
      }
      beg_id = end_id;beg_qpos = j;
      ab_band = yi & MM_SEED_BAND_MODE_MASK;
    }
    int qlen = tseq.size() - beg_qpos;
    std::vector<res_t> t_res = poa(para, DAG, beg_id, 1, rid, tseq.c_str() + j, qlen, mpool, ab_band);
    res.insert(res.end(), t_res.begin(), t_res.end());
  }
  else {
    int n = anchors.n;
    std::vector<std::vector<res_t>> res_v;
    res_v.resize(n + 1);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n + 1; i++) {
      int tid = omp_get_thread_num();
      int beg_id, end_id, start_qpos, qlen, q_span = para->k, end_qpos;
      beg_id = i == 0 ? 0 : DAG->cons_pos_to_id[(int)anchors.a[i - 1].x];
      end_id = i == n ? 1 : DAG->cons_pos_to_id[(int)anchors.a[i].x - q_span + 1];
      ab_band = i == 0 ? 1 : (anchors.a[i - 1].y & MM_SEED_BAND_MODE_MASK);
      end_qpos = i == n ? tseq.size() : (int)anchors.a[i].y - q_span + 1;
      start_qpos = i == 0 ? 0 : (int)anchors.a[i - 1].y + 1;
      qlen = end_qpos - start_qpos;
      std::vector<res_t> ret = poa(para, DAG, beg_id, end_id, rid, tseq.c_str() + start_qpos, qlen, &mpool[tid], ab_band);
      if (i < n) {
        for (int k = 0; k < q_span - 1; k++) {
          int end_tpos = (int)anchors.a[i].x - q_span + 1;
          int cur_id = DAG->cons_pos_to_id[end_tpos + k];
          const node_t &cur = DAG->node[cur_id];
          ret.emplace_back(res_t(cur.id, cur.base));
        }
      }
      res_v[i] = ret;
    }
    for (size_t i = 0; i < res_v.size(); i++) {
      res.insert(res.end(), res_v[i].begin(), res_v[i].end());
    }
  }

  // free anchors
  kfree(mm->km, anchors.a);
  return res;
}

// std::vector<res_t> POA(para_t* para, const graph& DAG, const std::string& seq) {
//   int n = DAG.node.size(), m = seq.size();
//   assert(seq[0] == para->m - 1);
//   const std::vector<node_t>& node = DAG.node;const std::vector<int>& rank = DAG.rank;
//   std::vector<int> mat = para->mat;
//   int match = para->match, mismatch = para->mismatch, para_m = para->m, e1 = para->gap_ext1, o1 = para->gap_open1 + e1;
//   // std::cout << mat << " " << mis << " " << o1 << " " << e1 << "\n";
//   std::vector<std::vector<int>> M(n, std::vector<int>(m + 1, NEG_INF)), D(M), I(M);
//   M[0][0] = 0;
//   // std::cout << "n:" << n << " " << "m:" << m << "\n";

//   for (int i = 1; i < n; i++) {
//     const node_t& cur = node[rank[i]];
//     // if (i % 100 == 0) std::cout << i << "\n";
//     for (int k = 0; k < cur.in.size(); k++) {
//       int p = node[cur.in[k]].rank; // rank
//       const node_t& pre = node[cur.in[k]];
//       for (int j = 1; j <= m; j++) {
//         if (j - 1 >= 0) M[i][j] = std::max(M[i][j], M[p][j - 1] + (pre.base == seq[j - 1] ? match : mismatch)); // qsource mat[pre.base * para_m + seq[j - 1]]
//         D[i][j] = std::max({ D[i][j], D[p][j] + e1, M[p][j] + o1 });    // dsource
//       }
//     }
//     for (int j = 1; j <= m; j++) {
//       if (j - 1 >= 0) I[i][j] = std::max(I[i][j - 1] + e1, M[i][j - 1] + o1); // isource
//       M[i][j] = std::max({ M[i][j], D[i][j], I[i][j] });  // three source
//     }
//   }
//   // std::cout << M[n - 1][m] << "\n";
//   // std::cout << "M:\n";
//   // for (int i = 0; i < n; i++) {
//   //   for (int j = 0; j <= m; j++) {
//   //     std::cout << M[i][j] << " \n"[j == m];
//   //   }
//   // }
//   // std::cout << "D:\n";
//   // for (int i = 0; i < n; i++) {
//   //   for (int j = 0; j <= m; j++) {
//   //     std::cout << D[i][j] << " \n"[j == m];
//   //   }
//   // }
//   // std::cout << "I:\n";
//   // for (int i = 0; i < n; i++) {
//   //   for (int j = 0; j <= m; j++) {
//   //     std::cout << I[i][j] << " \n"[j == m];
//   //   }
//   // }
//   int i = n - 1, j = m;
//   std::vector<res_t> res;
//   char op = 'M';
//   // std::cerr << "finsh align" << "\n";
//   while (i > 0 || j > 0) {
//     // dsource
//     if (op == 'M') {
//       if (M[i][j] == D[i][j]) op = 'D';
//       if (M[i][j] == I[i][j]) op = 'I';
//     }
//     // std::cerr << op << " " << i << " " << j << "\n";
//     if (op == 'M') {
//       const node_t& cur = node[rank[i]];
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         if (j - 1 >= 0 && M[i][j] == M[p][j - 1] + (pre.base == seq[j - 1] ? match : mismatch)) {
//           if (pre.base == seq[j - 1]) {
//             // std::cout << "M";
//             res.emplace_back(res_t(pre.id, pre.base));
//           }
//           else {
//             // std::cout << "X";
//             const node_t& par = node[pre.par_id]; // dsu.find par
//             if (par.aligned_node[seq[j - 1]] != -1) {
//               res.emplace_back(res_t(par.aligned_node[seq[j - 1]], seq[j - 1]));
//             }
//             else res.emplace_back(res_t(-1, seq[j - 1], pre.par_id));
//           }
//           i = p, j--;
//           break;
//         }
//       }
//       continue;
//     }
//     if (op == 'I') {
//       if (j - 1 >= 0) {
//         // std::cout << "I";
//         res.emplace_back(res_t(-1, seq[j - 1]));
//         if (I[i][j] == I[i][j - 1] + e1) {
//           j--;
//           continue;
//         }
//         if (I[i][j] == M[i][j - 1] + o1) {
//           op = 'M';
//           j--;
//           continue;
//         }
//       }
//     }
//     if (op == 'D') {
//       const node_t& cur = node[rank[i]];
//       // std::cerr << M[i][j] << " " << D[i][j] << "\n";
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         if (D[i][j] == D[p][j] + e1) {
//           i = p;
//           break;
//         }
//         if (D[i][j] == M[p][j] + o1) {
//           op = 'M';
//           i = p;
//           break;
//         }
//       }
//       continue;
//     }
//   }
//   // std::cout << "sorce:" << M[n - 1][m] << "\n";
//   return res;
// };

// std::vector<res_t> abPOA(const para_t* para, const graph* DAG, const minimizer_t* mm, int rid, const std::string& _seq, aligned_buff_t* mpool) {
//   // std::chrono::microseconds total_part1(0);
//   // std::chrono::microseconds total_part2(0);
//   // std::chrono::microseconds total_part3(0);
//   // auto start1 = std::chrono::high_resolution_clock::now();

//   int n = DAG->node.size(), m = _seq.size() + 1;
//   std::string seq;
//   seq += char26_table['N'];
//   // seq += node[beg_id].base;
//   for (int j = 0; j < _seq.size(); j++) {
//     seq += char26_table[_seq[j]];
//   }
//   assert(seq[0] == para->m - 1);
//   std::vector<res_t> res;
//   if (n <= 2) {
//     res.emplace_back(res_t(0, seq[0]));
//     for (int j = 1; j < m; j++) {
//       res.emplace_back(res_t(-1, seq[j]));
//     }
//     return res;
//   }
//   const std::vector<node_t>& node = DAG->node;const std::vector<int>& rank = DAG->rank;
//   std::vector<int> mat = para->mat;
//   int match = para->match, mismatch = para->mismatch, para_m = para->m, e1 = para->gap_ext1, o1 = para->gap_open1 + e1;
//   simd_reg MATCH = _mm256_set1_epi32(match), MISMATCH = _mm256_set1_epi32(mismatch), E1 = _mm256_set1_epi32(e1), O1 = _mm256_set1_epi32(o1), Neg_inf = _mm256_set1_epi32(NEG_INF);
//   // std::cout << mat << " " << mis << " " << o1 << " " << e1 << "\n";
//   int col_size = (m + 1 + simd_width - 1) / simd_width * simd_width;
//   seq.append(col_size - m, char26_table['N']);
//   // std::cerr << seq.size() << " " << col_size << "\n";
//   assert(col_size % simd_width == 0);
//   std::vector<int> Ms(n, m + 1), Me(n); // index is rank  [Ms[i], Me[i]]
//   std::vector<int> Bs(n), Be(n);  // index is rank    [Bs[i], Be[i] - 1)  Be[i] - 1 's block is Neg_inf is designed for M direciton dp
//   std::vector<int> Mp(n), Sl(n), Pl(n, m), Pr(n), Ol(n), Or(n), Ow(n);  // index is rank
//   size_t mtx_size = 0;
//   int w = para->b;
//   if (para->f) w += m / para->f;

//   size_t sum = 0;
//   for (int i = 0; i < n; i++) {
//     // Ms[i] = 0, Me[i] = m;
//     if (para->f > 0 && !para->enable_seeding) {
//       Ms[i] = std::max(0, std::min(DAG->hlen[i], m - DAG->tlen[i]) - w), Me[i] = std::min(m, std::max(DAG->hlen[i], m - DAG->tlen[i]) + w);
//     }
//     else {
//       Ms[i] = 0, Me[i] = m;
//     }
//     Bs[i] = Ms[i] / simd_width, Be[i] = Me[i] / simd_width + 2; // [block_s,block_e)
//     mtx_size += (Be[i] - Bs[i]) * simd_width;
//     // sum += Me[i] - Ms[i] + 1;
//   }
//   void* buff = nullptr;
//   // ===== 2. 对齐内存分配 =====
//   // if ((ret = posix_memalign(&buff, SIMDTotalBytes, 3 * mtx_size * sizeof(int))) != 0) {
//   // std::cerr << "ac_mtx_size:" << 1ll * n * m << "\n";
//   // std::cerr << "mtx_size:" << sum << "\n";
//   // std::cerr << m << " " << sum * 1.0 / n << "\n";
//   // return std::vector<res_t>();
//   // std::cerr << buff << "\n";
//   if (mpool != nullptr) mpool->alloc_aligned(&buff, SIMD_BYTES, 3 * mtx_size * sizeof(int));
//   else alloc_aligned(&buff, SIMD_BYTES, 3 * mtx_size * sizeof(int)); // malloc
//   std::vector<int*> M(n), D(n), I(n);
//   M[0] = (int*)buff;
//   D[0] = M[0] + mtx_size;
//   I[0] = D[0] + mtx_size;

//   for (int i = 1; i < n; i++) {
//     int offset = (Be[i - 1] - Bs[i - 1]) * simd_width;
//     M[i] = M[i - 1] + offset;
//     D[i] = D[i - 1] + offset;
//     I[i] = I[i - 1] + offset;
//   }
//   // std::cerr << "n:" << n << " " << w << "\n";
//   // std::vector<int> R = DAG->calculateR(); // index is rank
//   // std::cerr << "R:" << node[1].rank << " " << R[29650] << "\n";
//   // Ms[0] = Me[0] = 0;

//   // int tbs = std::max(0, std::min(DAG->hlen[0], m - DAG->tlen[0]) - w), tbe = std::min(m, std::max(DAG->hlen[0], m - DAG->tlen[0]) + w);
//   // Bs[0] = tbs, Be[0] = tbe;

//   // return std::vector<res_t>();
//   // int block_s = Bs[0] / reg_size, block_e = Be[0] / reg_size + 1; // [block_s,block_e)
//   // for (int i = 0; i < n; i++) {
//   //   Ms[i] = std::max(0, std::min(DAG->hlen[i], m - DAG->tlen[i]) - w), Me[i] = std::min(m, std::max(DAG->hlen[i], m - DAG->tlen[i]) + w);
//   //   Bs[i] = Ms[i] / reg_size, Be[i] = Me[i] / reg_size + reserved_reg_size; // [block_s,block_e)
//   // }
//   // std::cerr << w << "\n";
//   if (para->f > 0 && para->enable_seeding) {
//     Ms[0] = std::max(0, std::min({ Pl[0], DAG->hlen[0] + Ol[0], m - DAG->tlen[0] }) - w), Me[0] = std::min(m, std::max({ Pr[0], DAG->hlen[0] + Or[0], m - DAG->tlen[0] }) + w);
//     Bs[0] = Ms[0] / simd_width, Be[0] = Me[0] / simd_width + 2; // [block_s,block_e)
//   }
//   for (int bid = Bs[0]; bid < Be[0]; bid++) {
//     // memset(M, 0xc0, col_size * sizeof(int));//M[0] = NEG_INF
//     // memset(D, 0xc0, col_size * sizeof(int));//D[0] = NEG_INF
//     int* M_i = M[0];
//     int* D_i = D[0];
//     _mm256_store_si256((simd_reg*)(M_i + bid * simd_width), Neg_inf);
//     _mm256_store_si256((simd_reg*)(D_i + bid * simd_width), Neg_inf);
//   }
//   M[0][0] = 0;
//   // if (block_s - 1 >= 0) _mm256_store_si256((reg*)(M + (block_s - 1) * reg_size), Neg_inf);
//   // if (block_e < col_size / reg_size) _mm256_store_si256((reg*)(M + block_e * reg_size), Neg_inf);
//   // memset(M, 0xc0, col_size * sizeof(int));//M[0] = NEG_INF
//   // memset(D, 0xc0, col_size * sizeof(int));//D[0] = NEG_INF
//   // memset(buff, 0xc0, 3 * mtx_size * sizeof(int)); // 初始化为-INF 0xc0c0c0c0
//   // for (int k = 0; k < node[0].out.size(); k++) {
//   //   int suc = node[node[0].out[k]].rank;
//   //   Ms[suc] = std::min(Ms[suc], 0 + 1), Me[suc] = std::max(Me[suc], 0 + 1);
//   // }
//   // auto end1 = std::chrono::high_resolution_clock::now();
//   // total_part1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
//   int tot = 0;
//   // std::cerr << "node:" << node.size() << "\n";
//   // std::cerr << "path len:" << DAG->hlen[node[1].rank] << " " << DAG->tlen[node[0].rank] << "\n";
//   // std::cerr << "m:" << m << "\n";
//   // int mm_h_idx = mm->mm_h[rid];
//   int adaptive_cd = 0;
//   int max = 0;
//   int offset_band = 0;
//   for (int i = 1; i < n; i++) {
//     // std::cerr << i << " " << n << "\n";
//     adaptive_cd--;
//     const node_t& cur = node[rank[i]];
//     int* M_i = M[i];
//     int* D_i = D[i];
//     // updata Me[i], MS[i], Be[i], Bs[i] by offset
//     // tot += cur.in.size();
//     // Bs[i] = std::max(0, std::min(Ms[i], m - R[i]) - w), Be[i] = std::min(m, std::max(Me[i], m - R[i]) + w); // [Bs,Be]
//     // int tbs = std::max(0, std::min(DAG->hlen[i], m - DAG->tlen[i]) - w), tbe = std::min(m, std::max(DAG->hlen[i], m - DAG->tlen[i]) + w);
//     // Bs[i] = tbs, Be[i] = tbe;

//     // int tbs = std::max(0, std::min(DAG->hmin[i], m - DAG->tmax[i]) - w), tbe = std::min(m, std::max(DAG->hmax[i], m - DAG->tmin[i]) + w);
//     // std::cerr << m << " " << Bs[i] << " " << Be[i] << "\n";
//     // block_s = Bs[i] / reg_size, block_e = Be[i] / reg_size + 1; // [block_s,block_e)
//     // std::cerr << "B:" << m << " " << w << " " << tbs << " " << tbe << '\n';

//     if (para->f > 0 && para->ab_band) {  // adptive band
//       // int pmid = (Pl[i] + Pr[i]) / 2;
//       // int tms = std::min({ Pl[i], DAG->hlen[i] + Ol[i], m - DAG->tlen[i] }), tme = std::max({ Pr[i], DAG->hlen[i] + Or[i], m - DAG->tlen[i] });
//       // int len = std::max(pmid - tms + 1, tme - pmid);
//       // tms = std::max(0, pmid - len - w - Ow[i]), tme = std::min(m, pmid + len + w + Ow[i]);
//       // Ms[i] = tms, Me[i] = tme;
//       // int tw = para->b + (m - (DAG->hlen[i] + Ol[i])) / para->f;
//       Ms[i] = std::max(0, std::min({ Pl[i], DAG->hlen[i] + Ol[i], m - DAG->tlen[i] }) - w - Ow[i]), Me[i] = std::min(m, std::max({ Pr[i], DAG->hlen[i] + Or[i], m - DAG->tlen[i] }) + w + Ow[i]);

//       // int tbs = Ms[i] / reg_size, tbe = Me[i] / reg_size + 2;
//       // // assert(tbe - tbs <= Be[i] - Bs[i]);
//       // if (tbe - tbs > Be[i] - Bs[i]) {
//       //   std::cerr << i << " " << tbe - tbs << " " << Be[i] - Bs[i] << "\n";
//       //   std::cerr << "adptive band segmentation fault " << "\n";
//       //   exit(0);
//       // }
//       Bs[i] = Ms[i] / simd_width, Be[i] = Me[i] / simd_width + 2; // [block_s,block_e)
//     }

//     int block_num = Be[i] - Bs[i] - 1; // not contain Be[i] - 1 's block
//     sum += Me[i] - Ms[i] + 1;
//     auto start2 = std::chrono::high_resolution_clock::now();
//     for (int k = 0; k < cur.in.size(); k++) {
//       int p = node[cur.in[k]].rank; // rank
//       const node_t& pre = node[cur.in[k]];
//       simd_reg PRE_BASE = _mm256_set1_epi32(pre.base);
//       int prev = NEG_INF;
//       char prech = char26_table['N'];
//       int* M_p = M[p];
//       int* D_p = D[p];
//       for (int bid = 0; bid < block_num; bid++) { // SIMD
//         int j = bid * simd_width;
//         int acBid = (Bs[i] + bid);
//         int acj = acBid * simd_width;
//         simd_reg Mij = k == 0 ? Neg_inf : _mm256_load_si256((simd_reg*)(M_i + j));
//         reg Dij = k == 0 ? Neg_inf : _mm256_load_si256((reg*)(D_i + j));
//         reg SEQ; // seq[j - 1]
//         int pj = calj(acj, Bs[p]);
//         // std::cerr << "debug" << pj << "\n";
//         if (!((std::max(0, acBid - 1) < Bs[pre.rank]) || (acBid > Be[pre.rank] - 2))) { //Bs[pre.rank] <= j - 1 && j <= Be[pre.rank]
//           // std::cerr << "debug: " << bid << " " << Bs[pre.rank] / reg_size << " " << Be[pre.rank] / reg_size << "\n";
//           if (acj == 0) { // 特殊处理 j=0 的情况
//             alignas(32) int tmp[8] = { prech,seq[0], seq[1], seq[2],seq[3], seq[4], seq[5], seq[6] };
//             SEQ = _mm256_load_si256((reg*)tmp);
//           }
//           else {  // 直接加载连续内存 (高效)
//             const __m128i seq_chunk = _mm_loadu_si128((const __m128i*)(seq.c_str() + acj - 1));
//             SEQ = _mm256_cvtepi8_epi32(seq_chunk);  // 符号扩展到32位
//           }
//           // M[i][j] = std::max(M[i][j], M[p][j - 1] + mat[pre.base * para_m + seq[j - 1]]); // qsource
//           reg TMP = prev == NEG_INF ? _mm256_setr_epi32(prev, M_p[pj], M_p[pj + 1], M_p[pj + 2], M_p[pj + 3], M_p[pj + 4], M_p[pj + 5], M_p[pj + 6]) : _mm256_loadu_si256((reg*)(M_p + pj - 1)); //M[p][j - 1]

//           reg mask = _mm256_cmpeq_epi32(PRE_BASE, SEQ);
//           // pre.base == seq[j - 1] ? match : mismatch
//           TMP = _mm256_add_epi32(TMP, _mm256_blendv_epi8(MISMATCH, MATCH, mask));
//           Mij = _mm256_max_epi32(Mij, TMP); //std::max(M[i][j], M[p][j - 1] + mat[pre.base * para_m + seq[j - 1]]); 
//         }
//         if (Bs[pre.rank] <= acBid && acBid < Be[pre.rank] - 1) { // Bs[pre.rank] <= j && j <= Be[pre.rank]
//           // D[i][j] = std::max({ D[i][j], D[p][j] + e1, M[p][j] + o1 });    // dsource
//           reg TMP = _mm256_load_si256((reg*)(D_p + pj));
//           Dij = _mm256_max_epi32(Dij, _mm256_add_epi32(TMP, E1)); // max(D[i][j], D[p][j] + e1)
//           TMP = _mm256_load_si256((reg*)(M_p + pj));
//           Dij = _mm256_max_epi32(Dij, _mm256_add_epi32(TMP, O1)); // max(D[i][j], M[p][j] + o1)
//         }
//         _mm256_store_si256((reg*)(M_i + j), Mij); // M[i][j] = max
//         _mm256_store_si256((reg*)(D_i + j), Dij); // D[i][j] = max
//         prev = M_p[pj + reg_size - 1]; // pre = M[p][j +reg_size - 1]
//         // prech = seq[j + reg_size - 1];// prech =seq[j + reg_size - 1]

//       }
//     }
//     // Be[i] - 1
//     // if (block_s - 1 >= 0) _mm256_store_si256((reg*)(M_i + (block_s - 1) * reg_size), Neg_inf);
//     _mm256_store_si256((reg*)(M_i + block_num * reg_size), Neg_inf);
//     _mm256_store_si256((reg*)(D_i + block_num * reg_size), Neg_inf);

//     // if (i % 100 == 0) std::cout << i << "\n";
//     // auto end2 = std::chrono::high_resolution_clock::now();
//     // total_part2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
//     // auto start3 = std::chrono::high_resolution_clock::now();
//     // I[i][0] = NEG_INF;
//     int* I_i = I[i];
//     // I_i[std::max(1, Bs[i] * reg_size) - 1] = M_i[std::max(1, Bs[i] * reg_size) - 1] = NEG_INF;
//     // int max_pos = -1;
//     I_i[0] = NEG_INF;
//     M_i[0] = std::max({ M_i[0], D_i[0], I_i[0] });
//     for (int j = 1; j < block_num * reg_size; j++) { // block_num * reg_size
//       I_i[j] = std::max(I_i[j - 1] + e1, M_i[j - 1] + o1); // isource
//       M_i[j] = std::max({ M_i[j], D_i[j], I_i[j] });  // three source
//     }
//     _mm256_store_si256((reg*)(I_i + block_num * reg_size), Neg_inf);

//     if (para->f > 0 && para->ab_band) {
//       int max_j = 0;
//       for (int j = 1; j < block_num* reg_size; j++) { // block_num * reg_size
//         max_j = M_i[j] > M_i[max_j] ? j : max_j;
//       }
//       int max_acj = Bs[i] * reg_size + max_j, max_slp = 0;
//       bool isMatch = false;
//       Mp[i] = max_acj;
//       if (cur.base == seq[max_acj]) {
//         for (int k = 0; k < cur.in.size(); k++) {
//           int p = node[cur.in[k]].rank; // rank
//           max_slp = Sl[p] > max_slp ? Sl[p] : max_slp;
//           if (Mp[p] + 1 == Mp[i]) {
//             Sl[i] = std::max(Sl[i], Sl[p] + 1);
//             isMatch = true;
//           }
//         }
//       }
//       if (!isMatch && max_slp < 30) {
//         // w += max_slp;
//         Ow[i] += max_slp + 1;
//         Ow[i] = std::min(Ow[i], 5 * w);
//       }
//       if (Sl[i] >= 30) {
//         // offset_band = max_acj - DAG->hlen[i];
//         Ol[i] = max_acj - DAG->hlen[i];
//         Or[i] = max_acj - DAG->hlen[i];

//         // sum_offset += max_acj - DAG->hlen[i];
//         // max = std::max(max, offset_band);
//         // std::cerr << para->b + (m - max_acj + 1) / para->f << "\n";
//         // w = para->b + m / para->f;
//         Ow[i] = 0;
//         tot++;
//         // Sl[i] = 0;
//         for (int k = 0; k < cur.out.size(); k++) {
//           int suc = node[cur.out[k]].rank;
//           Ol[suc] = Ol[i], Or[suc] = Or[i];
//         }
//       }
//       for (int k = 0; k < cur.out.size(); k++) {
//         int suc = node[cur.out[k]].rank;
//         Ol[suc] = std::min(Ol[suc], Ol[i]), Or[suc] = std::max(Or[suc], Or[i]);
//       }

//       for (int k = 0; k < cur.out.size(); k++) {
//         int suc = node[cur.out[k]].rank;
//         Pl[suc] = std::min(Pl[suc], max_acj), Pr[suc] = std::max(Pr[suc], max_acj + 1);
//         Ow[suc] = Ow[i] ? std::max(Ow[suc], Ow[i]) : Ow[i];
//         // int tms = std::min({ Pl[suc], DAG->hlen[suc] + Ol[suc], m - DAG->tlen[suc] }), tme = std::max({ Pr[suc], DAG->hlen[suc] + Or[suc], m - DAG->tlen[suc] });
//         // int tbs = tms / reg_size, tbe = tme / reg_size;
//         // Pr[suc] = std::max(Pr[suc], max_acj + (max_acj - tbs * reg_size));
//         // Pl[suc] = std::min(Pl[suc], max_acj - (tbe * reg_size - max_acj));
//       }
//     }

//     // if (para->enable_seeding && adaptive_cd <= 0) {
//     //   int max_j = 0;
//     //   for (int j = 1; j < block_num* reg_size; j++) { // block_num * reg_size
//     //     max_j = M_i[j] > M_i[max_j] ? j : max_j;
//     //   }
//     //   int max_acj = Bs[i] * reg_size + max_j;
//     //   int is_M_OP = -1;
//     //   for (int k = 0; k < cur.in.size(); k++) {
//     //     int p = node[cur.in[k]].rank; // rank
//     //     const node_t& pre = node[cur.in[k]];
//     //     if (pre.base != seq[max_acj - 1]) continue;
//     //     // M
//     //     int pj = calj(max_acj, Bs[p]);
//     //     // if (ans == 5114 && pj - 1 >= 0) 
//     //       // std::cerr << M[p][pj - 1] << " " << (pre.base == seq[acj - 1] ? match : mismatch) << "\n";
//     //     if (pj - 1 >= 0 && M[i][max_j] == M[p][pj - 1] + match) {
//     //       is_M_OP = k;
//     //       break;
//     //     }
//     //   }
//     //   if (is_M_OP != -1 && i < n - 1) {
//     //     // match minimizer
//     //     // std::cerr << cur.id << " " << char256_table[cur.base] << " " << "M source" << "\n";
//     //     mm128_t seed = mm->find_mm(rid, max_acj - 2);  // mm_h_idx will be updated to the next idx needed match 
//     //     bool find = 0;
//     //     if (((seed.y >> 1) & 0x7FFFFFFF) == max_acj - 2) {
//     //       bool isAnchored = false;
//     //       // try max_similary mm
//     //       const node_t& pre = node[cur.in[is_M_OP]];
//     //       mm128_t seed_g = mm->match_mm(seed.x, mm->max_sim[rid]);
//     //       // std::cerr << pre.id << " " << max_acj - 2 << " " << seed.x << "\n";
//     //       // std::cerr << pre.getPos(mm->rid_to_ord[mm->max_sim[rid]]) << " " << 
//     //       // std::cerr << "\n";
//     //       if (seed_g.x == seed.x && pre.getPos(mm->rid_to_ord[mm->max_sim[rid]]) >= ((seed_g.y >> 1) & 0x7FFFFFFF)) { //&& pos[rid][(seed_g.y >> 1) & 0x7FFFFFFF)] >= cur.id
//     //         // std::cerr << "anchored" << "\n";
//     //         // std::cerr << pre.getPos(mm->rid_to_ord[mm->max_sim[rid]]) << " " << max_acj - 2 << "\n";
//     //         offset_band = max_acj - DAG->hlen[i];
//     //         // sum_offset += max_acj - DAG->hlen[i];
//     //         max = std::max(max, offset_band);
//     //         // std::cerr << para->b + (m - max_acj + 1) / para->f << "\n";
//     //         w = para->b + (m - max_acj + 1) / para->f;
//     //         adaptive_cd = para->k * 5;
//     //         find = 1;
//     //         tot++;
//     //       }
//     //       // for() cur.ids;
//     //       // update offset
//     //       //

//     //     }
//     //     if (!find) {
//     //       adaptive_cd = para->w - 1;
//     //     }

//     //   }

//     // }

//     // if (debug) {
//     //   // for (int j = std::max(1, block_s * reg_size); j < block_e * reg_size; j++) {
//     //   //   std::cerr << i << " " << j << " " << M_i[j] << "\n";
//     //   // }
//     //   return std::vector<res_t>();
//     // }
//     // auto end3 = std::chrono::high_resolution_clock::now();
//     // total_part3 += std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);
//     // for (int k = 0; k < cur.out.size(); k++) {
//     //   int suc = node[cur.out[k]].rank;
//     //   Ms[suc] = std::min(Ms[suc], max_pos + 1), Me[suc] = std::max(Me[suc], max_pos + 1);
//     // }
//   }
//   // std::cerr << sum * 1.0 / (1ll * n * m) << "\n";
//   // std::cerr << "anchor num:" << tot << "\n";
//   // std::cerr << "sum_offset" << max << "\n";
//   // std::cerr << "avg pre size:" << tot / n << "\n";
//   // std::cerr << "部分1总耗时: " << total_part1.count() / 1000 << " ms\n";
//   // std::cerr << "部分2总耗时: " << total_part2.count() / 1000 << " ms\n";
//   // std::cerr << "部分3总耗时: " << total_part3.count() / 1000 << " ms\n";
//   int i = n - 1, acj = m;
//   int ans = M[i][calj(acj, Bs[i])];
//   // std::cout << "M:\n";
//   // for (int i = 0; i < n; i++) {
//   //   std::cout << i << ":" << "\n";
//   //   for (int j = 0; j < Be[i] * reg_size; j++) {
//   //     std::cout << M[i][j] << " ";
//   //   }
//   //   std::cout << "\n";
//   // }
//   // std::cout << "D:\n";
//   // for (int i = 0; i < n; i++) {
//   //   std::cout << "Be:" << Bs[i] << " " << Be[i] << "\n";
//   //   for (int j = 0; j < Be[i] * reg_size; j++) {
//   //     std::cout << D[i][j] << " ";
//   //   }
//   //   std::cout << "\n";

//   // }
//   // std::cout << "I:\n";
//   // for (int i = 0; i < n; i++) {
//   //   std::cout << "Be:" << Bs[i] << " " << Be[i] << "\n";
//   //   for (int j = 0; j < Be[i] * reg_size; j++) {
//   //     std::cout << I[i][j] << " ";
//   //   }
//   //   std::cout << "\n";
//   // }
//   // return std::vector<res_t>();
//   // if (3 * mtx_size == 5208) {
//   //   std::cout << "M:\n";
//   //   for (int i = 0; i < n; i++) {
//   //     std::cout << "Be:" << Bs[i] << " " << Be[i] << "\n";
//   //     for (int j = 0; j < col_size; j++) {
//   //       std::cout << M[i * col_size + j] << " \n"[j + 1 == col_size];
//   //     }
//   //   }
//   //   std::cout << "D:\n";
//   //   for (int i = 0; i < n; i++) {
//   //     std::cout << "Be:" << Bs[i] << " " << Be[i] << "\n";
//   //     for (int j = 0; j < col_size; j++) {
//   //       std::cout << D[i * col_size + j] << " \n"[j + 1 == col_size];
//   //     }
//   //   }
//   //   std::cout << "I:\n";
//   //   for (int i = 0; i < n; i++) {
//   //     std::cout << "Be:" << Bs[i] << " " << Be[i] << "\n";
//   //     for (int j = 0; j < col_size; j++) {
//   //       std::cout << I[i * col_size + j] << " \n"[j + 1 == col_size];
//   //     }
//   //   }
//   // }
//   // if (3 * mtx_size == 5208) std::cerr << _seq.size() << " " << seq.size() << "\n";

//   // return res;
//   int j = calj(acj, Bs[i]);
//   int op = ALL_OP;
//   // std::cerr << rid << "\n";
//   // std::cerr << i << " " << j << " " << M[i][calj(acj, Bs[i])] << "\n";
//   // std::cerr << "finsh align" << "\n";
//   // std::cerr << "finish" << "\n";
//   while (i > 0 || acj > 0) {
//     // dsource
//     j = calj(acj, Bs[i]);
//     if (acj > (Be[i] - 1) * reg_size || acj < Bs[i] * reg_size) {
//       std::cerr << "l:" << Bs[i] * reg_size << " " << "r:" << (Be[i] - 1) * reg_size << "\n";
//       std::cerr << acj << "\n";
//       exit(0);
//     }
//     // std::cerr << op << " " << i << " " << j << "\n";
//     // if (j < 0 || j >(Be[i] - 1) * reg_size) {
//     //   std::cerr << ans << "\n";
//     //   std::cerr << "run time error" << "\n";
//     //   exit(0);
//     // }
//     // std::cerr << M[i][j] << " " << D[i][j] << " " << I[i][j] << "\n";
//     // if (M[i][j] < NEG_INF / 2) {
//     //   std::cerr << "run time error" << "\n";
//     //   exit(0);
//     // }
//     // if (ans == 5114 && op == 'M') {
//       // std::cerr << M[i][j] << "\n";
//       // std::cerr << D[i][j] << "\n";
//       // std::cerr << I[i][j] << "\n";
//     // }
//     // if (ans == 5114) std::cerr << op << " " << i << " " << acj << "\n";
//     // if (ans == 5114 && op == 'M' && i == 6840 && acj == 1182) {
//     //   // exit(0);
//     // }
//     // if (3 * mtx_size == 3237179904) {
//     //   if (op == 'M') {
//     //     std::cerr << M[i * col_size + j] << "\n";
//     //   }
//     //   else if (op == 'D')
//     //   {
//     //     std::cerr << D[i * col_size + j] << "\n";
//     //   }
//     //   if (i == 0 && j == 51) {
//     //     return res;
//     //   }
//     //   if (i == 0) exit(1);
//     // }
//     // std::cerr << op << "\n";
//     // if (op == ALL_OP) {  // M
//     //   // std::cerr << op << " " << i << " " << j << " ";
//     //   // std::cerr << M[i][j] << " " << D[i][j] << " " << I[i][j] << "\n";
//     //   // if (i == 15932 && j == 205) exit(0);
//     //   const node_t& cur = node[rank[i]];
//     //   int bk = -1;
//     //   for (int k = 0; k < cur.in.size(); k++) {
//     //     int p = node[cur.in[k]].rank; // rank
//     //     const node_t& pre = node[cur.in[k]];
//     //     if (pre.base != seq[acj - 1]) continue;
//     //     // M
//     //     int pj = calj(acj, Bs[p]);
//     //     // if (ans == 5114 && pj - 1 >= 0) 
//     //       // std::cerr << M[p][pj - 1] << " " << (pre.base == seq[acj - 1] ? match : mismatch) << "\n";
//     //     if (pj - 1 >= 0 && M[i][j] == M[p][pj - 1] + match && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
//     //       bk = k;
//     //       // break;
//     //     }
//     //   }
//     //   // std::cerr << bk << "\n";
//     //   // assert(bk != -1);
//     //   if (bk != -1) {
//     //     // std::cerr << "M";
//     //     op = ALL_OP;
//     //     int p = node[cur.in[bk]].rank; // rank
//     //     const node_t& pre = node[cur.in[bk]];
//     //     if (pre.base == seq[acj - 1]) {
//     //       // std::cout << "M";
//     //       res.emplace_back(res_t(pre.id, pre.base));
//     //     }
//     //     else {
//     //       // std::cout << "X";
//     //       const node_t& par = node[pre.par_id]; // dsu.find par
//     //       if (par.aligned_node[seq[acj - 1]] != -1) {
//     //         res.emplace_back(res_t(par.aligned_node[seq[acj - 1]], seq[acj - 1]));
//     //       }
//     //       else res.emplace_back(res_t(-1, seq[acj - 1], pre.par_id));
//     //     }
//     //     i = p, acj--;
//     //     continue;
//     //   }
//     // }
//     if (op & M_OP && acj > 0) {  // M
//       // std::cerr << op << " " << i << " " << j << " ";
//       // std::cerr << M[i][j] << " " << D[i][j] << " " << I[i][j] << "\n";
//       // if (i == 15932 && j == 205) exit(0);
//       const node_t& cur = node[rank[i]];
//       int bk = -1;
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         if (pre.base != seq[acj - 1]) continue;
//         // M
//         int pj = calj(acj, Bs[p]);
//         // if (ans == 5114 && pj - 1 >= 0) 
//           // std::cerr << M[p][pj - 1] << " " << (pre.base == seq[acj - 1] ? match : mismatch) << "\n";
//         int block_num = Be[p] - Bs[p] - 1; // not contain Be[i] - 1 's block
//         if (pj - 1 >= 0 && pj - 1 < block_num * reg_size && M[i][j] == M[p][pj - 1] + (pre.base == seq[acj - 1] ? match : mismatch) && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
//           bk = k;
//           // break;
//         }
//       }
//       // std::cerr << bk << "\n";
//       // assert(bk != -1);
//       if (bk != -1 && cur.in_weight[bk] >= cur.ind / 10) {  // backtrack based on the normal sample
//         // std::cerr << "M";
//         op = ALL_OP;
//         int p = node[cur.in[bk]].rank; // rank
//         const node_t& pre = node[cur.in[bk]];
//         if (pre.base == seq[acj - 1]) {
//           // std::cout << "M";
//           res.emplace_back(res_t(pre.id, pre.base));
//         }
//         else {
//           // std::cout << "X";
//           const node_t& par = node[pre.par_id]; // dsu.find par
//           if (par.aligned_node[seq[acj - 1]] != -1) {
//             res.emplace_back(res_t(par.aligned_node[seq[acj - 1]], seq[acj - 1]));
//           }
//           else res.emplace_back(res_t(-1, seq[acj - 1], pre.par_id));
//         }
//         i = p, acj--;
//         continue;
//       }
//     }
//     if (op & D_OP) {
//       if (op == D_OP || M[i][j] == D[i][j]) {
//         // std::cerr << M[i][j] << " " << D[i][j] << "\n";
//         const node_t& cur = node[rank[i]];
//         // std::cerr << M[i * col_size + j] << " " << D[i * col_size + j] << "\n";
//         int bk = -1;char bop;
//         for (int k = 0; k < cur.in.size(); k++) {
//           int p = node[cur.in[k]].rank; // rank
//           const node_t& pre = node[cur.in[k]];
//           if (Bs[pre.rank] <= acj / reg_size && acj / reg_size < Be[pre.rank] - 1) {
//             int pj = calj(acj, Bs[p]);
//             if (D[i][j] == M[p][pj] + o1 && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
//               bk = k;
//               bop = M_OP | I_OP;
//               // break;
//             }
//           }
//         }
//         if (bk == -1) {
//           for (int k = 0; k < cur.in.size(); k++) {
//             int p = node[cur.in[k]].rank; // rank
//             const node_t& pre = node[cur.in[k]];
//             if (Bs[pre.rank] <= acj / reg_size && acj / reg_size < Be[pre.rank] - 1) {
//               int pj = calj(acj, Bs[p]);
//               if (D[i][j] == D[p][pj] + e1 && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
//                 bk = k;
//                 bop = D_OP;
//                 // break;
//               }
//             }
//           }
//         }
//         // std::cerr << bk << "\n";
//         if (bk != -1) {
//           i = node[cur.in[bk]].rank;
//           op = bop;
//           continue;
//         }
//       }
//     }
//     if (op & I_OP && acj > 0) {
//       const node_t& cur = node[rank[i]];
//       if (j - 1 >= 0) {
//         if (op == I_OP || M[i][j] == I[i][j]) {
//           // std::cerr << "I";
//           if (I[i][j] == M[i][j - 1] + o1) {
//             res.emplace_back(res_t(-1, seq[acj - 1]));
//             op = M_OP | D_OP;
//             acj--;
//             continue;
//           }
//           if (I[i][j] == I[i][j - 1] + e1) {
//             res.emplace_back(res_t(-1, seq[acj - 1]));
//             op = I_OP;
//             acj--;
//             continue;
//           }
//         }
//       }
//     }
//     if (op & M_OP && acj > 0) {  // MX
//       // std::cerr << op << " " << i << " " << j << " ";
//       // std::cerr << M[i][j] << " " << D[i][j] << " " << I[i][j] << "\n";
//       // if (i == 15932 && j == 205) exit(0);
//       const node_t& cur = node[rank[i]];
//       int bk = -1;
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         // if (pre.base != seq[acj - 1]) continue;
//         // M
//         int pj = calj(acj, Bs[p]);
//         // if (M[i][j] == 17586 && pj - 1 >= 0)
//         //   std::cerr << M[p][pj - 1] << " " << (pre.base == seq[acj - 1] ? match : mismatch) << "\n";
//         int block_num = Be[p] - Bs[p] - 1; // not contain Be[i] - 1 's block
//         if (pj - 1 >= 0 && pj - 1 < block_num * reg_size && M[i][j] == M[p][pj - 1] + (pre.base == seq[acj - 1] ? match : mismatch) && (bk == -1 || cur.in_weight[k] > cur.in_weight[bk])) {
//           bk = k;
//           // break;
//         }
//       }
//       // std::cerr << bk << "\n";
//       // assert(bk != -1);
//       if (bk != -1) {
//         // std::cerr << "M";
//         op = ALL_OP;
//         int p = node[cur.in[bk]].rank; // rank
//         const node_t& pre = node[cur.in[bk]];
//         if (pre.base == seq[acj - 1]) {
//           // std::cout << "M";
//           res.emplace_back(res_t(pre.id, pre.base));
//         }
//         else {
//           // std::cout << "X";
//           const node_t& par = node[pre.par_id]; // dsu.find par
//           if (par.aligned_node[seq[acj - 1]] != -1) {
//             res.emplace_back(res_t(par.aligned_node[seq[acj - 1]], seq[acj - 1]));
//           }
//           else res.emplace_back(res_t(-1, seq[acj - 1], pre.par_id));
//         }
//         i = p, acj--;
//         continue;
//       }
//     }

//     // if (op &) {
//     //   if (M[i][j] == D[i][j]) op = 'D';
//     //   if (op == 'M' && M[i][j] == I[i][j]) op = 'I';
//     // }
//     // std::cerr << Bs[i] * reg_size << " " << acj << " " << Be[i] * reg_size << "\n";
//     // std::cerr << M[i][j] << " " << D[i][j] << " " << I[i][j] << "\n";
//     // std::cerr << i << " " << n - 1 << " " << acj << " " << m << "\n";
//     std::cerr << " backtrack error" << "\n";
//     exit(0);
//     // if (op == 'M') {
//     //   if (M[i][j] == I[i][j]) op = 'I';
//     //   if (op == 'M' && M[i][j] == D[i][j]) op = 'D';
//     // }

//   }
//   // std::cerr << "\n";
//   // if (acj > 0) {
//   //   std::cerr << "acj:" << acj << "\n";
//   //   exit(0);
//   // }
//   // std::cerr << "finish" << "\n";
//   // std::cout << "sorce:" << M[n - 1][m] << "\n";
//   if (mpool == nullptr) free_aligned(buff);
//   std::reverse(res.begin(), res.end());
//   return res;
// }

// std::vector<res_t> POA_SIMD_ORIGIN(const para_t* para, const graph* DAG, const std::string& _seq, aligned_buff_t* mpool) {
//   std::chrono::microseconds total_part1(0);
//   std::chrono::microseconds total_part2(0);
//   std::chrono::microseconds total_part3(0);
//   auto start1 = std::chrono::high_resolution_clock::now();

//   int n = DAG->node.size(), m = _seq.size();
//   assert(_seq[0] == para->m - 1);
//   const std::vector<node_t>& node = DAG->node;const std::vector<int>& rank = DAG->rank;
//   std::vector<int> mat = para->mat;
//   int match = para->match, mismatch = para->mismatch, para_m = para->m, e1 = para->gap_ext1, o1 = para->gap_open1 + e1;
//   reg MATCH = _mm256_set1_epi32(match), MISMATCH = _mm256_set1_epi32(mismatch), E1 = _mm256_set1_epi32(e1), O1 = _mm256_set1_epi32(o1), Neg_inf = _mm256_set1_epi32(NEG_INF);
//   // std::cout << mat << " " << mis << " " << o1 << " " << e1 << "\n";
//   int reg_size = SIMDTotalBytes / sizeof(int); // number of int in reg
//   int col_size = (m + 1 + reg_size - 1) / reg_size * reg_size;
//   const size_t mtx_size = n * col_size;
//   assert(col_size % reg_size == 0);
//   int block_num = col_size / reg_size; // block_num
//   std::string seq;seq.resize(col_size);
//   for (int i = 0; i < col_size; i++) {
//     // if (i < m) seq[(i % block_size) * reg_size + (i / block_size)] = _seq[i];
//     // else seq[(i % block_size) * reg_size + (i / block_size)] = char26_table['N'];
//     seq[(i % block_num) * reg_size + (i / block_num)] = i < m ? _seq[i] : char26_table['N'];
//   }
//   // std::cerr << seq.size() << " " << col_size << "\n";
//   void* buff = nullptr;
//   // ===== 2. 对齐内存分配 =====
//   // if ((ret = posix_memalign(&buff, SIMDTotalBytes, 3 * mtx_size * sizeof(int))) != 0) {
//   // std::cerr << 3 * mtx_size << "\n";
//   // return std::vector<res_t>();
//   // std::cerr << buff << "\n";
//   if (mpool != nullptr) mpool->alloc_aligned(&buff, SIMDTotalBytes, 3 * mtx_size * sizeof(int));
//   else alloc_aligned(&buff, SIMDTotalBytes, 3 * mtx_size * sizeof(int)); // malloc
//   int* M = (int*)buff;
//   int* D = M + mtx_size;
//   int* I = D + mtx_size;
//   memset(M, 0xc0, col_size * sizeof(int));//M[0] = NEG_INF
//   memset(D, 0xc0, col_size * sizeof(int));//D[0] = NEG_INF
//   // memset(buff, 0xc0, 3 * mtx_size * sizeof(int)); // 初始化为-INF 0xc0c0c0c0
//   M[0 * col_size + 0] = 0;
//   auto end1 = std::chrono::high_resolution_clock::now();
//   total_part1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
//   int tot = 0;
//   int offset = (block_num - 1) * reg_size;
//   for (int i = 1; i < n; i++) {
//     const node_t& cur = node[rank[i]];
//     int* M_i = M + i * col_size;
//     int* D_i = D + i * col_size;
//     tot += cur.in.size();
//     auto start2 = std::chrono::high_resolution_clock::now();
//     for (int k = 0; k < cur.in.size(); k++) {
//       int p = node[cur.in[k]].rank; // rank
//       const node_t& pre = node[cur.in[k]];
//       reg PRE_BASE = _mm256_set1_epi32(pre.base);
//       // j == 0;
//       reg Mij = k == 0 ? Neg_inf : _mm256_load_si256((reg*)(M_i));
//       reg SEQ; // seq[j - 1]
//       // 特殊处理 j=0 的情况
//       alignas(32) int tmp[8] = { char26_table['N'], seq[offset + 0], seq[offset + 1], seq[offset + 2],seq[offset + 3], seq[offset + 4], seq[offset + 5], seq[offset + 6] };
//       SEQ = _mm256_load_si256((reg*)tmp);

//       int* M_p = M + p * col_size;
//       // M[i][j] = std::max(M[i][j], M[p][j - 1] + mat[pre.base * para_m + seq[j - 1]]); // qsource
//       reg TMP = _mm256_setr_epi32(NEG_INF, M_p[offset + 0], M_p[offset + 1], M_p[offset + 2], M_p[offset + 3], M_p[offset + 4], M_p[offset + 5], M_p[offset + 6]); //M[p][j - 1]
//       reg mask = _mm256_cmpeq_epi32(PRE_BASE, SEQ);
//       // pre.base == seq[j - 1] ? match : mismatch
//       TMP = _mm256_add_epi32(TMP, _mm256_blendv_epi8(MISMATCH, MATCH, mask));
//       Mij = _mm256_max_epi32(Mij, TMP); //std::max(M[i][j], M[p][j - 1] + mat[pre.base * para_m + seq[j - 1]]); 
//       _mm256_store_si256((reg*)(M_i), Mij); // M[i][j] = max

//       for (int j = reg_size; j < col_size; j += reg_size) { // SIMD Msource
//         Mij = k == 0 ? Neg_inf : _mm256_load_si256((reg*)(M_i + j));
//         SEQ; // seq[j - 1]
//         // 直接加载连续内存 (高效)
//         const __m128i seq_chunk = _mm_loadu_si128((const __m128i*)(seq.c_str() + j - reg_size));
//         SEQ = _mm256_cvtepi8_epi32(seq_chunk);  // 符号扩展到32位

//         int* M_p = M + p * col_size;
//         // M[i][j] = std::max(M[i][j], M[p][j - 1] + mat[pre.base * para_m + seq[j - 1]]); // qsource
//         TMP = _mm256_load_si256((reg*)(M_p + j - reg_size)); //M[p][j - 1]

//         mask = _mm256_cmpeq_epi32(PRE_BASE, SEQ);
//         // pre.base == seq[j - 1] ? match : mismatch
//         TMP = _mm256_add_epi32(TMP, _mm256_blendv_epi8(MISMATCH, MATCH, mask));
//         Mij = _mm256_max_epi32(Mij, TMP); //std::max(M[i][j], M[p][j - 1] + mat[pre.base * para_m + seq[j - 1]]); 
//         _mm256_store_si256((reg*)(M_i + j), Mij); // M[i][j] = max
//       }
//       for (int j = 0; j < col_size; j += reg_size) { // SIMD Dsource
//         reg Dij = k == 0 ? Neg_inf : _mm256_load_si256((reg*)(D_i + j));
//         int* M_p = M + p * col_size;
//         int* D_p = D + p * col_size;
//         // D[i][j] = std::max({ D[i][j], D[p][j] + e1, M[p][j] + o1 });    // dsource
//         TMP = _mm256_load_si256((reg*)(D_p + j));
//         Dij = _mm256_max_epi32(Dij, _mm256_add_epi32(TMP, E1)); // max(D[i][j], D[p][j] + e1)
//         TMP = _mm256_load_si256((reg*)(M_p + j));
//         Dij = _mm256_max_epi32(Dij, _mm256_add_epi32(TMP, O1)); // max(D[i][j], M[p][j] + o1)
//         _mm256_store_si256((reg*)(D_i + j), Dij); // D[i][j] = max
//       }
//     }

//     auto end2 = std::chrono::high_resolution_clock::now();
//     total_part2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
//     auto start3 = std::chrono::high_resolution_clock::now();
//     // SIMD fsource
//     // I[i][0] = NEG_INF;
//     int* I_i = I + i * col_size;
//     I_i[0] = NEG_INF;
//     for (int j = 1; j < reg_size; j++) {
//       // I[i][j] = std::max(I[i][j - 1] + e1, M[i][j - 1] + o1)
//       I_i[j] = std::max(I_i[j - 1] + block_num * e1, M_i[j - 1 + offset] + o1); // isource
//       M_i[j] = std::max({ M_i[j], D_i[j], I_i[j] });  // three source
//     }
//     for (int j = reg_size; j < col_size; j += reg_size) {
//       // I[i][j] = std::max(I[i][j - 1] + e1, M[i][j - 1] + o1)
//       reg TMP1 = _mm256_load_si256((reg*)(I_i + j - reg_size));
//       TMP1 = _mm256_add_epi32(TMP1, E1);
//       reg TMP2 = _mm256_load_si256((reg*)(M_i + j - reg_size));
//       TMP2 = _mm256_add_epi32(TMP2, O1);
//       _mm256_store_si256((reg*)(I_i + j), _mm256_max_epi32(TMP1, TMP2)); // I[i][j] = max

//       // M[i][j] = std::max({ M[i][j], D[i][j], I[i][j] });  // three source
//       reg Mij = _mm256_load_si256((reg*)(M_i + j));
//       TMP1 = _mm256_load_si256((reg*)(D_i + j));
//       TMP2 = _mm256_load_si256((reg*)(I_i + j));
//       Mij = _mm256_max_epi32(Mij, _mm256_max_epi32(TMP1, TMP2));
//       _mm256_store_si256((reg*)(M_i + j), Mij); // M[i][j] = max
//     }
//     // active f loop
//     I_i[1] = std::max(I_i[offset] + e1, M_i[offset] + o1); // isource
//     M_i[1] = std::max({ M_i[1], D_i[1], I_i[1] });  // three source
//     for (int j = 2; j < reg_size; j++) {
//       // I[i][j] = std::max(I[i][j - 1] + e1, M[i][j - 1] + o1)
//       I_i[j] = std::max(I_i[j], I_i[j - 1] + block_num * e1); // isource
//       M_i[j] = std::max(M_i[j], I_i[j]);  // three source
//     }
//     for (int j = reg_size; j < col_size; j += reg_size) {
//       // I[i][j] = std::max(I[i][j], I[i][j - 1] + e1)
//       reg Iij = _mm256_load_si256((reg*)(I_i + j));
//       reg TMP1 = _mm256_load_si256((reg*)(I_i + j - reg_size)); // I[i][j - 1]
//       TMP1 = _mm256_add_epi32(TMP1, E1); //I[i][j - 1] + e1
//       _mm256_store_si256((reg*)(I_i + j), _mm256_max_epi32(Iij, TMP1)); // I[i][j] = max

//       // M[i][j] = std::max(M[i][j], I[i][j]);  // three source
//       reg Mij = _mm256_load_si256((reg*)(M_i + j));
//       TMP1 = _mm256_load_si256((reg*)(I_i + j));
//       _mm256_store_si256((reg*)(M_i + j), _mm256_max_epi32(Mij, TMP1)); // M[i][j] = max
//     }
//     auto end3 = std::chrono::high_resolution_clock::now();
//     total_part3 += std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);
//   }
//   // std::cerr << "avg pre size:" << tot / n << "\n";
//   // std::cerr << "部分1总耗时: " << total_part1.count() / 1000 << " ms\n";
//   // std::cerr << "部分2总耗时: " << total_part2.count() / 1000 << " ms\n";
//   // std::cerr << "部分3总耗时: " << total_part3.count() / 1000 << " ms\n";
//   // std::cout << M[(n - 1) * col_size + m] << "\n";
//   // std::cout << "M:\n";
//   // for (int i = 0; i < n; i++) {
//   //   for (int j = 0; j < col_size; j++) {
//   //     std::cout << M[i * col_size + j] << " \n"[j + 1 == col_size];
//   //   }
//   // }
//   // std::cout << "D:\n";
//   // for (int i = 0; i < n; i++) {
//   //   for (int j = 0; j < col_size; j++) {
//   //     std::cout << D[i * col_size + j] << " \n"[j + 1 == col_size];
//   //   }
//   // }
//   // std::cout << "I:\n";
//   // for (int i = 0; i < n; i++) {
//   //   for (int j = 0; j < col_size; j++) {
//   //     std::cout << I[i * col_size + j] << " \n"[j + 1 == col_size];
//   //   }
//   // }
//   // std::cerr << "finish dp" << "\n";
//   int i = n - 1, j = (m % block_num) * reg_size + (m / block_num); // j = m
//   // std::cerr << M[i * col_size + j] << "\n";
//   std::vector<res_t> res;
//   // return res;
//   char op = 'M';
//   // std::cerr << "finsh align" << "\n";
//   while (i > 0 || j > 0) {
//     // std::cerr << "op: " << op << "\n";
//     // std::cerr << "i: " << i << "\n";
//     // std::cerr << "j: " << j << "\n";
//     int* M_i = M + i * col_size;
//     int* D_i = D + i * col_size;
//     int* I_i = I + i * col_size;
//     int hit = 0;
//     // dsource
//     // std::cerr << op << " " << i << " " << j << "\n";
//     if (op == 'M') {
//       if (M_i[j] == D_i[j]) op = 'D';
//       if (M_i[j] == I_i[j]) op = 'I';
//     }
//     if (op == 'M') {
//       const node_t& cur = node[rank[i]];
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         int* M_p = M + p * col_size;
//         if (j - reg_size >= 0) {
//           if (pre.base == seq[j - reg_size] && M_i[j] == M_p[j - reg_size] + match) {
//             // std::cout << "M";
//             res.emplace_back(res_t(pre.id, pre.base));
//             i = p, j -= reg_size;
//             hit = 1;
//             break;
//           }
//         }
//         else {
//           if (pre.base == seq[j - 1 + offset] && M_i[j] == M_p[j - 1 + offset] + match) {
//             // std::cout << "M";
//             res.emplace_back(res_t(pre.id, pre.base));
//             i = p, j = j - 1 + offset;
//             hit = 1;
//             break;
//           }
//         }
//       }

//       for (int k = 0; !hit && k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         int* M_p = M + p * col_size;
//         if (j - reg_size >= 0) {
//           if (pre.base != seq[j - reg_size] && M_i[j] == M_p[j - reg_size] + mismatch) {
//             const node_t& par = node[pre.par_id]; // dsu.find par
//             if (par.aligned_node[seq[j - reg_size]] != -1) {
//               res.emplace_back(res_t(par.aligned_node[seq[j - reg_size]], seq[j - reg_size]));
//             }
//             else res.emplace_back(res_t(-1, seq[j - reg_size], pre.par_id));
//             i = p, j -= reg_size;
//             hit = 1;
//             break;
//           }
//         }
//         else {
//           if (pre.base != seq[j - 1 + offset] && M_i[j] == M_p[j - 1 + offset] + mismatch) {
//             // std::cout << "X";
//             const node_t& par = node[pre.par_id]; // dsu.find par
//             if (par.aligned_node[seq[j - 1 + offset]] != -1) {
//               res.emplace_back(res_t(par.aligned_node[seq[j - 1 + offset]], seq[j - 1 + offset]));
//             }
//             else res.emplace_back(res_t(-1, seq[j - 1 + offset], pre.par_id));
//             i = p, j = j - 1 + offset;
//             hit = 1;
//             break;
//           }
//         }
//       }
//       if (hit) continue;
//     }
//     if (op == 'I') {
//       if (j - reg_size >= 0) {
//         // std::cout << "I";
//         res.emplace_back(res_t(-1, seq[j - reg_size]));
//         if (I_i[j] == I_i[j - reg_size] + e1) {
//           j -= reg_size;
//           continue;
//         }
//         if (I_i[j] == M_i[j - reg_size] + o1) {
//           op = 'M';
//           j -= reg_size;
//           continue;
//         }
//       }
//       else {
//         res.emplace_back(res_t(-1, seq[j - 1 + offset]));
//         if (I_i[j] == I_i[j - 1 + offset] + e1) {
//           j = j - 1 + offset;
//           continue;
//         }
//         if (I_i[j] == M_i[j - 1 + offset] + o1) {
//           op = 'M';
//           j = j - 1 + offset;
//           continue;
//         }
//       }
//     }
//     if (op == 'D') {
//       const node_t& cur = node[rank[i]];
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         int* D_p = D + p * col_size;
//         int* M_p = M + p * col_size;
//         if (D_i[j] == D_p[j] + e1) {
//           i = p;
//           break;
//         }
//       }
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         int* D_p = D + p * col_size;
//         int* M_p = M + p * col_size;
//         if (D_i[j] == M_p[j] + o1) {
//           op = 'M';
//           i = p;
//           break;
//         }
//       }
//       continue;
//     }
//   }
//   // std::cerr << "finish POA:" << "\n";
//   if (mpool == nullptr) free_aligned(buff);
//   return res;
// }

// std::vector<res_t> POA_SIMD(para_t* para, const graph& DAG, const std::string& _seq, aligned_buff_t* mpool) {
//   std::chrono::microseconds total_part1(0);
//   std::chrono::microseconds total_part2(0);
//   std::chrono::microseconds total_part3(0);
//   auto start1 = std::chrono::high_resolution_clock::now();

//   int n = DAG.node.size(), m = _seq.size();
//   std::string seq = _seq;
//   assert(seq[0] == para->m - 1);
//   const std::vector<node_t>& node = DAG.node;const std::vector<int>& rank = DAG.rank;
//   std::vector<int> mat = para->mat;
//   int match = para->match, mismatch = para->mismatch, para_m = para->m, e1 = para->gap_ext1, o1 = para->gap_open1 + e1;
//   reg MATCH = _mm256_set1_epi32(match), MISMATCH = _mm256_set1_epi32(mismatch), E1 = _mm256_set1_epi32(e1), O1 = _mm256_set1_epi32(o1), Neg_inf = _mm256_set1_epi32(NEG_INF);
//   // std::cout << mat << " " << mis << " " << o1 << " " << e1 << "\n";
//   int reg_size = SIMDTotalBytes / sizeof(int); // number of int in reg
//   int col_size = (m + 1 + reg_size - 1) / reg_size * reg_size;
//   seq.append(col_size - m, char26_table['N']);
//   // std::cerr << seq.size() << " " << col_size << "\n";
//   assert(col_size % reg_size == 0);
//   const size_t mtx_size = n * col_size;
//   size_t max_in_size = 0;
//   for (int i = 0; i < n; i++) {
//     const node_t& cur = node[rank[i]];
//     max_in_size = std::max(max_in_size, cur.in.size());
//   }
//   void* buff = nullptr;
//   // ===== 2. 对齐内存分配 =====
//   // if ((ret = posix_memalign(&buff, SIMDTotalBytes, 3 * mtx_size * sizeof(int))) != 0) {
//   std::cerr << 3 * mtx_size << "\n";
//   // return std::vector<res_t>();
//   // std::cerr << buff << "\n";
//   if (mpool != nullptr) mpool->alloc_aligned(&buff, SIMDTotalBytes, 3 * mtx_size * sizeof(int));
//   else alloc_aligned(&buff, SIMDTotalBytes, 3 * mtx_size * sizeof(int)); // malloc
//   int* M = (int*)buff;
//   int* D = M + mtx_size;
//   int* I = D + mtx_size;
//   memset(M, 0xc0, col_size * sizeof(int));//M[0] = NEG_INF
//   memset(D, 0xc0, col_size * sizeof(int));//D[0] = NEG_INF
//   // memset(buff, 0xc0, 3 * mtx_size * sizeof(int)); // 初始化为-INF 0xc0c0c0c0
//   M[0 * col_size + 0] = 0;
//   auto end1 = std::chrono::high_resolution_clock::now();
//   total_part1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
//   int tot = 0;
//   std::vector<int> p(max_in_size), prev(max_in_size);  // 自动初始化值为0 
//   reg* PRE_BASE = nullptr;
//   alloc_aligned((void**)&PRE_BASE, SIMDTotalBytes, max_in_size * sizeof(reg));
//   for (int i = 1; i < n; i++) {
//     const node_t& cur = node[rank[i]];
//     int* M_i = M + i * col_size;
//     int* D_i = D + i * col_size;
//     tot += cur.in.size();
//     auto start2 = std::chrono::high_resolution_clock::now();
//     char prech = char26_table['N'];
//     for (int k = 0; k < cur.in.size(); k++) {
//       p[k] = node[cur.in[k]].rank; // rank
//       prev[k] = NEG_INF;
//       const node_t& pre = node[cur.in[k]];
//       PRE_BASE[k] = _mm256_set1_epi32(pre.base);
//     }
//     for (int j = 0; j < col_size; j += reg_size) { // SIMD
//       reg Mij = Neg_inf;
//       reg Dij = Neg_inf;
//       reg SEQ;
//       if (j == 0) {
//         // 特殊处理 j=0 的情况
//         alignas(32) int tmp[8] = {
//             prech,
//             seq[0], seq[1], seq[2],
//             seq[3], seq[4], seq[5], seq[6]
//         };
//         SEQ = _mm256_load_si256((reg*)tmp);
//       }
//       else {
//         // 直接加载连续内存 (高效)
//         const __m128i seq_chunk = _mm_loadu_si128((const __m128i*)(seq.c_str() + j - 1));
//         SEQ = _mm256_cvtepi8_epi32(seq_chunk);  // 符号扩展到32位
//       }
//       for (int k = 0; k < cur.in.size(); k++) {
//         int* M_p = M + p[k] * col_size;
//         int* D_p = D + p[k] * col_size;
//         // M[i][j] = std::max(M[i][j], M[p][j - 1] + mat[pre.base * para_m + seq[j - 1]]); // qsource
//     //j - 1;
//         reg TMP = j == 0 ? _mm256_setr_epi32(prev[k], M_p[j], M_p[j + 1], M_p[j + 2], M_p[j + 3], M_p[j + 4], M_p[j + 5], M_p[j + 6]) : _mm256_loadu_si256((reg*)(M_p + j - 1)); //M[p][j - 1]

//         reg mask = _mm256_cmpeq_epi32(PRE_BASE[k], SEQ);
//         // pre.base == seq[j - 1] ? match : mismatch
//         TMP = _mm256_add_epi32(TMP, _mm256_blendv_epi8(MISMATCH, MATCH, mask));
//         Mij = _mm256_max_epi32(Mij, TMP); //std::max(M[i][j], M[p][j - 1] + mat[pre.base * para_m + seq[j - 1]]); 

//         // D[i][j] = std::max({ D[i][j], D[p][j] + e1, M[p][j] + o1 });    // dsource
//         TMP = _mm256_load_si256((reg*)(D_p + j));
//         Dij = _mm256_max_epi32(Dij, _mm256_add_epi32(TMP, E1)); // max(D[i][j], D[p][j] + e1)
//         TMP = _mm256_load_si256((reg*)(M_p + j));
//         Dij = _mm256_max_epi32(Dij, _mm256_add_epi32(TMP, O1)); // max(D[i][j], M[p][j] + o1)

//         prev[k] = M_p[j + reg_size - 1]; // pre = M[p][j +reg_size - 1]
//       }
//       prech = seq[j + reg_size - 1];// prech =seq[j + reg_size - 1]
//       _mm256_store_si256((reg*)(M_i + j), Mij); // M[i][j] = max
//       _mm256_store_si256((reg*)(D_i + j), Dij); // D[i][j] = max
//     }
//     // if (i % 100 == 0) std::cout << i << "\n";

//     auto end2 = std::chrono::high_resolution_clock::now();
//     total_part2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
//     auto start3 = std::chrono::high_resolution_clock::now();
//     // I[i][0] = NEG_INF;
//     int* I_i = I + i * col_size;
//     I_i[0] = NEG_INF;
//     for (int j = 1; j <= m; j++) {
//       I_i[j] = std::max(I_i[j - 1] + e1, M_i[j - 1] + o1); // isource
//       M_i[j] = std::max({ M_i[j], D_i[j], I_i[j] });  // three source
//     }
//     auto end3 = std::chrono::high_resolution_clock::now();
//     total_part3 += std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);
//   }
//   std::cerr << "avg pre size:" << tot / n << "\n";
//   std::cerr << "部分1总耗时: " << total_part1.count() / 1000 << " ms\n";
//   std::cerr << "部分2总耗时: " << total_part2.count() / 1000 << " ms\n";
//   std::cerr << "部分3总耗时: " << total_part3.count() / 1000 << " ms\n";
//   std::cerr << M[(n - 1) * col_size + m] << "\n";
//   std::cout << "M:\n";
//   for (int i = 0; i < n; i++) {
//     for (int j = 0; j < col_size; j++) {
//       std::cout << M[i * col_size + j] << " \n"[j + 1 == col_size];
//     }
//   }
//   std::cout << "D:\n";
//   for (int i = 0; i < n; i++) {
//     for (int j = 0; j < col_size; j++) {
//       std::cout << D[i * col_size + j] << " \n"[j + 1 == col_size];
//     }
//   }
//   std::cout << "I:\n";
//   for (int i = 0; i < n; i++) {
//     for (int j = 0; j < col_size; j++) {
//       std::cout << I[i * col_size + j] << " \n"[j + 1 == col_size];
//     }
//   }
//   // std::cerr << "finish" << "\n";
//   int i = n - 1, j = m;
//   std::vector<res_t> res;
//   char op = 'M';
//   // std::cerr << "finsh align" << "\n";
//   while (i > 0 || j > 0) {
//     // dsource
//     std::cerr << op << " " << i << " " << j << "\n";
//     if (op == 'M') {
//       if (M[i * col_size + j] == D[i * col_size + j]) op = 'D';
//       if (M[i * col_size + j] == I[i * col_size + j]) op = 'I';
//     }
//     // std::cerr << op << " " << i << " " << j << "\n";
//     if (op == 'M') {
//       const node_t& cur = node[rank[i]];
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         if (j - 1 >= 0 && M[i * col_size + j] == M[p * col_size + j - 1] + (pre.base == seq[j - 1] ? match : mismatch)) {
//           if (pre.base == seq[j - 1]) {
//             // std::cout << "M";
//             res.emplace_back(res_t(pre.id, pre.base));
//           }
//           else {
//             // std::cout << "X";
//             const node_t& par = node[pre.par_id]; // dsu.find par
//             if (par.aligned_node[seq[j - 1]] != -1) {
//               res.emplace_back(res_t(par.aligned_node[seq[j - 1]], seq[j - 1]));
//             }
//             else res.emplace_back(res_t(-1, seq[j - 1], pre.par_id));
//           }
//           i = p, j--;
//           break;
//         }
//       }
//       continue;
//     }
//     if (op == 'I') {
//       if (j - 1 >= 0) {
//         // std::cout << "I";
//         res.emplace_back(res_t(-1, seq[j - 1]));
//         if (I[i * col_size + j] == I[i * col_size + j - 1] + e1) {
//           j--;
//           continue;
//         }
//         if (I[i * col_size + j] == M[i * col_size + j - 1] + o1) {
//           op = 'M';
//           j--;
//           continue;
//         }
//       }
//     }
//     if (op == 'D') {
//       const node_t& cur = node[rank[i]];
//       // std::cerr << M[i][j] << " " << D[i][j] << "\n";
//       for (int k = 0; k < cur.in.size(); k++) {
//         int p = node[cur.in[k]].rank; // rank
//         const node_t& pre = node[cur.in[k]];
//         if (D[i * col_size + j] == D[p * col_size + j] + e1) {
//           i = p;
//           break;
//         }
//         if (D[i * col_size + j] == M[p * col_size + j] + o1) {
//           op = 'M';
//           i = p;
//           break;
//         }
//       }
//       continue;
//     }
//   }
//   // std::cout << "sorce:" << M[n - 1][m] << "\n";
//   free_aligned(PRE_BASE);
//   if (mpool == nullptr) free_aligned(buff);
//   return res;
// }
