"""轻量纯 Python BPE（Byte-Pair Encoding）分词器，用于中文类语料。

- learn(corpus, vocab_target)：词级（空格/换行切分）内做字符对合并，学 vocab+merges
- encode(text)/decode(ids)：按 learned merges 编码 / 解码
- save/load：持久化 vocab 与 merges

实现为「word 内增量合并」：维护每个 word 的相邻对计数 + 全局计数堆（lazy 删除），
避免每步 O(全文) 重扫；在几百万~上千万字符规模下足够快。
"""

import heapq
import json
import re
import sys

UNK = 1


class BPE:
    def __init__(self):
        self.itos = {}
        self.stoi = {}
        self.merges = {}      # (a, b) -> new_id

    # ---------- learn ----------
    def learn(self, corpus, vocab_target=12000, min_pair_count=2, word_limit=None):
        # 预处理：按空白/换行切成 word（保留空白为词内 token，保证解码可还原空格/换行）
        parts = re.split(r'(\s+)', corpus)
        # 字符初始词表
        chars = sorted(set(corpus))  # includes '\n',' ', and any visible chars
        self.stoi = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.itos = {v: k for k, v in self.stoi.items()}
        for c in chars:
            if c not in self.stoi:
                self.stoi[c] = len(self.stoi)
                self.itos[self.stoi[c]] = c

        words = []
        for p in parts:
            if p == '':
                continue
            words.append([self.stoi.get(c, UNK) for c in p])

        # 每 word 内相邻对计数（局部折叠）
        # word_pairs[wi] = dict[pair] = count ；global_pair_count[pair] += count
        import collections
        word_pair_counts = []
        global_count = collections.Counter()
        for wi, w in enumerate(words):
            local = collections.Counter()
            for a, b in zip(w, w[1:]):
                local[(a, b)] += 1
            word_pair_counts.append(local)
            for pair, c in local.items():
                global_count[pair] += c

        # 堆：(-count, pair)。lazy：弹出时用当前 global_count 校验
        heap = [(-c, p) for p, c in global_count.items() if c >= min_pair_count]
        heapq.heapify(heap)
        merges = []
        next_id = len(self.stoi)
        ITER = vocab_target - len(self.stoi)   # 还能合多少个
        print('start: chars=%d, pairs=%d, 目标新增 merges=%d' % (len(self.stoi), len(global_count), ITER), flush=True)

        made = 0
        while heap and made < ITER:
            negc, pair = heap[0]
            cur = global_count.get(pair, 0)
            if cur != -negc:
                heapq.heapreplace(heap, (-cur, pair))
                if cur < min_pair_count:
                    heapq.heappop(heap)
                continue
            heapq.heappop(heap)
            if cur < min_pair_count:
                continue
            # 合并
            new_id = next_id
            next_id += 1
            self.itos[new_id] = self.itos[pair[0]] + self.itos[pair[1]]
            self.stoi[self.itos[new_id]] = new_id
            merges.append(pair)
            self.merges[pair] = new_id
            # 更新受影响的 word 及其局部/全局计数
            for wi, w in enumerate(words):
                idxs = [i for i in range(len(w) - 1) if (w[i], w[i + 1]) == pair]
                if not idxs:
                    continue
                local = word_pair_counts[wi]
                # 先去掉旧的对计数
                for i in idxs:
                    self._dec(local, global_count, (w[i], w[i + 1]))
                    if i > 0:
                        self._dec(local, global_count, (w[i - 1], w[i]))
                    if i + 2 < len(w):
                        self._dec(local, global_count, (w[i + 1], w[i + 2]))
                # 生成新 word
                nw = []
                i = 0
                L = len(w)
                while i < L:
                    if i + 1 < L and (w[i], w[i + 1]) == pair:
                        nw.append(new_id)
                        i += 2
                    else:
                        nw.append(w[i])
                        i += 1
                # 加上新对计数
                for a, b in zip(nw, nw[1:]):
                    self._inc(local, global_count, (a, b))
                words[wi] = nw
            made += 1
            if made % 500 == 0:
                print('  merges=%d/%d next_id=%d' % (made, ITER, next_id), flush=True)

        self.vocab_size = len(self.stoi)
        return merges

    @staticmethod
    def _inc(local, global_count, pair):
        local[pair] += 1
        global_count[pair] += 1

    @staticmethod
    def _dec(local, global_count, pair):
        v = global_count[pair]
        if v <= 1:
            del global_count[pair]
        else:
            global_count[pair] = v - 1
        v2 = local[pair]
        if v2 <= 1:
            del local[pair]
        else:
            local[pair] = v2 - 1

    # ---------- encode/decode ----------
    def encode(self, text):
        out = []
        for p in re.split(r'(\s+)', text):
            if not p:
                continue
            ids = [self.stoi.get(c, UNK) for c in p]
            # 应用 merges：贪心左到右合并
            i = 0
            ids2 = []
            while i < len(ids):
                # 若当前 token 能与之前的合并，则不新开（BPE 是『能并就并』）
                ids2.append(ids[i])
                i += 1
                # 回溯：尽量把末尾两 token 合并
                while len(ids2) >= 2:
                    a, b = ids2[-2], ids2[-1]
                    t = self.merges.get((a, b))
                    if t is None:
                        break
                    ids2.pop(); ids2.pop(); ids2.append(t)
            out.extend(ids2)
        return out

    def decode(self, ids):
        return ''.join(self.itos.get(i, '�') for i in ids)

    # ---------- io ----------
    def save(self, path):
        json.dump({'itos': {str(k): v for k, v in self.itos.items()},
                   'merges': {f'{a},{b}': c for (a, b), c in self.merges.items()}},
                  open(path, 'w', encoding='utf-8'), ensure_ascii=False)

    def load(self, path):
        d = json.load(open(path, encoding='utf-8'))
        self.itos = {int(k): v for k, v in d['itos'].items()}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.merges = {tuple(map(int, k.split(','))): c for k, c in d['merges'].items()}
        self.vocab_size = len(self.itos)


if __name__ == '__main__':
    # 自测
    corpus = '这是一段中文测试文本。这是另一句测试。人工智能很厉害，机器学习是它的基础。'
    b = BPE()
    b.learn(corpus, vocab_target=30)
    print('vocab', b.vocab_size)
    ids = b.encode('这是另一句测试。')
    print('encoded', ids, '->', b.decode(ids))
    print('roundtrip ok:', b.decode(ids) == '这是另一句测试。')
