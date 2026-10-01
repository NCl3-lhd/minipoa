#include <iostream>
#include <algorithm>
#include <numeric>
#include <unistd.h>
#include <string>
#include "cxxopts.hpp"
#include "minipoa.h"
#include <omp.h>
// #include "kband.h"
// #include "timer.h"
// extern unsigned char nt4_table[256];
int main(int argc, char **argv) {
  // handle arg
  cxxopts::Options options("minipoa", "A minimizer-based method for fast and memory-efficient partial order alignment\nExamples:\n  Sequencing mode for consensus calling:\tminipoa input.fasta > output.fasta\n  MSA mode for multiple sequence alignment:\tminipoa input.fasta -S -r1 -t thread > output.fasta\n  Generate graph information in GFA format:\tminipoa input.fasta -S -r2 -t thread > output.gfa");

  options.add_options()
    ("input", "input path", cxxopts::value<std::string>())
    ("i,inc_fp", "incrementally align sequences to an existing graph", cxxopts::value<std::string>())
    ("m,mat_fp", "scoring matrix file path (requirement: N-N match score > N-other base score)", cxxopts::value<std::string>())
    ("M,match", "match score", cxxopts::value<int>()->default_value("2"))
    ("X,mismatch", "mismatch penalty", cxxopts::value<int>()->default_value("-4"))
    ("O,gap_open", "gap open penalty", cxxopts::value<int>()->default_value("-4"))
    ("E,gap_ext", "gap extension penalty", cxxopts::value<int>()->default_value("-2"))
    ("t,thread", "thread number", cxxopts::value<int>()->default_value("1"))
    ("b,band_b", "base band width (total band = b + seq_len / f)", cxxopts::value<int>()->default_value("100"))
    ("f,band_f", "band expansion factor (smaller = wider band)", cxxopts::value<int>()->default_value("40"))
    ("B,ab_band", "enable adaptive band", cxxopts::value<bool>()->default_value("false"))
    ("S,seeding", "enable minimizer-based seeding and anchoring", cxxopts::value<bool>()->default_value("false"))
    ("W,poa_w", "the minimum distance between adjacent anchors", cxxopts::value<int>()->default_value("20000"))
    ("k,k_mer", "k-mer length", cxxopts::value<int>()->default_value("19")) //19
    ("w,mm_w", "minimizer window length", cxxopts::value<int>()->default_value("10"))
    ("p,progressive_poa", "enable progressive POA guide tree", cxxopts::value<bool>()->default_value("false"))
    ("r,result", "result format (0-2). 0: consensus, 1: rc-msa, 2: gfa", cxxopts::value<int>()->default_value("0"))
    ("V,verbose", "verbose level (0-2). 0: none, 1: information, 2: debug [0]\n", cxxopts::value<int>()->default_value("0"))
    ("h,help", "Print usage")
    ("v,version", "Print version")
    ("mm-filter-ratio", "minimizer filter ratio for guide tree construction", cxxopts::value<float>()->default_value("0.25"))
    ;
  options.parse_positional({ "input" });
  std::string path;
  para_t *para = new para_t();
  try {
    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
      std::cout << options.help() << std::endl;
      return 0;
    }
    if (result.count("version")) {
      std::cout << "minipoa version 1.4.3" << std::endl;
      return 0;
    }
    if (result.count("input") == 0) {
      std::cout << options.help() << std::endl;
      return 1;
    }
    if (result.count("input"))
      path = result["input"].as<std::string>();
    if (result.count("mat_fp"))
      para->mat_fp = result["mat_fp"].as<std::string>();
    if (result.count("inc_fp"))
      para->inc_fp = result["inc_fp"].as<std::string>();
    para->poa_w = 0;
    if (result.count("poa_w"))
      para->poa_w = result["poa_w"].as<int>();
    para->match = result["match"].as<int>();
    para->mismatch = result["mismatch"].as<int>();
    para->gap_open1 = result["gap_open"].as<int>();
    para->gap_ext1 = result["gap_ext"].as<int>();
    para->b = result["band_b"].as<int>();
    para->f = result["band_f"].as<int>();
    para->ab_band = result["ab_band"].as<bool>();
    para->k = result["k_mer"].as<int>();
    para->mm_w = result["mm_w"].as<int>();
    para->bw = 1000;
    para->progressive_poa = result["progressive_poa"].as<bool>();
    para->mm_filter_ratio = result["mm-filter-ratio"].as<float>();
    para->enable_seeding = result["seeding"].as<bool>();
    para->thread = result["thread"].as<int>();
    para->result = result["result"].as<int>();
    para->verbose = result["verbose"].as<int>();

  }
  catch (const cxxopts::exceptions::exception &e)
  {
    std::cerr << "error parsing options: " << e.what() << std::endl;
    std::cout << options.help() << std::endl;
    return 1;
  }

  // handle para
  initPara(para);

  // handle input
  std::vector<seq_t> seqs;
  graph *DAG = new graph();
  DAG->init(para);
  std::string tmp_path = "minipoa_paths_" + std::to_string(getpid()) + ".tmp";
  DAG->tmp_path = tmp_path;
  PathWriter *writer = para->result ? new PathWriter(tmp_path.c_str()) : nullptr;

  if (!para->inc_fp.empty()) seqs = read_gfa(para, DAG, para->inc_fp.c_str(), writer);
  int exist_seq_num = seqs.size();
  try {
    if (!path.empty()) readFile(para, seqs, path.c_str());
    // std::cerr << exist_seq_num << " " << seqs.size() << " " << DAG->node.size() << "\n";
  }
  catch (const std::exception &e) {
    std::cerr << "error read file: " << e.what() << std::endl;
    return 1;
  }
  if (para->isRNA) char256_table[3] = 'U';
  // if (seqs.size() < 5000) para->progressive_poa |= 1;

  if (para->verbose && para->enable_seeding) std::cerr << "collect minimizer" << "\n";
  minimizer_t *mm = new minimizer_t(para, seqs);
  if (para->verbose && para->progressive_poa) std::cerr << "build guide tree" << "\n";
  if (para->inc_fp.empty() && para->progressive_poa) {
    if (para->verbose) std::cerr << "progressive" << "\n";
    mm->get_guide_tree(para);
  }
  const std::vector<int> &ord = mm->ord;

  int rid;
  if (para->verbose) std::cerr << "poa" << "\n";
  aligned_buff_t *mpool = new aligned_buff_t[para->thread];
  omp_set_num_threads(para->thread);
  for (size_t i = exist_seq_num; i < seqs.size(); i++) {  //seqs.size()
    rid = ord[i];
    if (para->verbose && i % 10 == 0) {
      std::cerr << "[" << i << "/" << seqs.size() << "]" << "\n";
    }
    if (para->verbose >= 2) std::cerr << "aligment" << "\n";
    std::vector<res_t> res = alignment(para, DAG, mm, rid, seqs[rid].seq.c_str(), seqs[rid].seq.size(), mpool);
    if (para->verbose >= 2) std::cerr << "add path" << "\n";
    DAG->add_path(para, rid, res, writer, 1);
    if (para->verbose >= 2) std::cerr << "topsort" << "\n";
    DAG->topsort(para, i + 1 == seqs.size());
  }
  if (para->result) {
    writer->close();
  }
  // Timer::instance().start("output");
  if (para->verbose) std::cerr << "out_put" << "\n";
  if (para->result == 0) DAG->output_consensus();
  else if (para->result == 1) DAG->output_rc_msa(para, mm->rid_to_ord, seqs);
  else if (para->result == 2) DAG->output_gfa(mm->rid_to_ord, seqs);
  // Timer::instance().stop("output");
  // Timer::instance().print();

  // delete
  if (para->result) unlink(tmp_path.c_str());
  delete writer;
  writer = nullptr;
  delete[] mpool;
  mpool = nullptr;  // 防止后续误用
  delete para;
  para = nullptr;  // 防止后续误用
  delete DAG;
  DAG = nullptr;  // 防止后续误用
  delete mm;
  mm = nullptr;  // 防止后续误用
  return 0;
}