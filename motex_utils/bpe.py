"""轻量纯 Python BPE（Byte-Pair Encoding）分词器，用于中文类语料。

- learn(corpus, vocab_target, extra_chars=None, byte_mode=True)：词级（空格/换行切分）内做字符对合并
  extra_chars：全语料高频单字保底（先入词表，覆盖 learn 样本之外的高频字）
  byte_mode：预置 256 个字节槽（id 4..259），编码未命中字符时按 UTF-8 拆字节（永不 UNK）
- encode(text)/decode(ids)：按 learned merges 编码 / 解码（byte_mode 下自动回退字节槽）
- save/load：持久化 vocab 与 merges（字节槽以 '<bXX>' 文本标记序列化）

实现为「word 内增量合并」：维护每个 word 的相邻对计数 + 全局计数堆（lazy 删除），
避免每步 O(全文) 重扫；在几百万~上千万字符规模下足够快。
"""

import array
import heapq
import json
import re
import sys

UNK = 1
BYTE_OFFSET = 4                  # 4 specials 之后给 256 字节槽
BYTE_SLOTS = 256


def _char_to_bytes(c):
    """单个字符 → UTF-8 字节序列（0-255 列表）"""
    return list(c.encode('utf-8'))


def _bytes_to_char(bs):
    return bytes(bs).decode('utf-8', errors='replace')


class BPE:
    def __init__(self):
        self.itos = {}
        self.stoi = {}
        self.merges = {}      # (a, b) -> new_id
        self.byte_mode = False

    # ---------- learn ----------
    def learn(self, corpus, vocab_target=12000, min_pair_count=2, word_limit=None,
              extra_chars=None, byte_mode=True):
        """byte_mode：预置字节槽；extra_chars：单字保底（未在样本中也入表）"""
        self.byte_mode = byte_mode
        parts = re.split(r'(\s+)', corpus)
        chars = sorted(set(corpus))          # 样本中出现字符
        if extra_chars:
            for c in extra_chars:
                if c not in chars:
                    chars.append(c)
            chars.sort()
        self.stoi = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.itos = {v: k for k, v in self.stoi.items()}
        if byte_mode:
            # 字节槽：<b00>..<bFF>（4..259）
            for b in range(BYTE_SLOTS):
                self.stoi[f'<b{b:02X}>'] = BYTE_OFFSET + b
                self.itos[BYTE_OFFSET + b] = f'<b{b:02X}>'
        for c in chars:
            if c not in self.stoi:
                self.stoi[c] = len(self.stoi)
                self.itos[self.stoi[c]] = c

        words = []
        # 内存纪律：learn 内部按固定 512 字符分块（中文无空格，整篇=1 词会内存爆炸），
        # 用 array('I') 存 id（4B/int，替代 Python int 对象 ~28B）
        CHUNK = 512
        for i in range(0, len(corpus), CHUNK):
            p = corpus[i:i + CHUNK]
            w = array.array('I')
            for c in p:
                if c in self.stoi:
                    w.append(self.stoi[c])
                elif byte_mode:
                    w.extend(BYTE_OFFSET + b for b in _char_to_bytes(c))   # 未命中 → 字节槽
                else:
                    w.append(UNK)
            words.append(w)

        # pair→words 候选索引（合并只处理相关词，避免全量扫描）
        import collections
        # 全局打包键计数（int 键替代 tuple，空间/哈希双省）+ pair→words 候选索引
        # 键 = a * PK_BASE + b（id 上限 2^17=131072，PK_BASE 取 200000 安全）
        PK_BASE = 200000
        global_count = collections.Counter()
        pair_words = collections.defaultdict(list)
        for wi, w in enumerate(words):
            for i in range(len(w) - 1):
                k = w[i] * PK_BASE + w[i + 1]
                global_count[k] += 1
                pair_words[k].append(wi)

        # 堆：(-count, key)。lazy：弹出时用当前 global_count 校验
        heap = [(-c, k) for k, c in global_count.items() if c >= min_pair_count]
        heapq.heapify(heap)
        merges = []
        next_id = len(self.stoi)
        ITER = vocab_target - len(self.stoi)   # 还能合多少个
        print('start: chars=%d, pairs=%d, 目标新增 merges=%d' % (len(self.stoi), len(global_count), ITER), flush=True)

        made = 0
        while heap and made < ITER:
            negc, key = heap[0]
            cur = global_count.get(key, 0)
            if cur != -negc:
                heapq.heapreplace(heap, (-cur, key))
                if cur < min_pair_count:
                    heapq.heappop(heap)
                continue
            heapq.heappop(heap)
            if cur < min_pair_count:
                continue
            a, b = divmod(key, PK_BASE)
            pair = (a, b)
            # 合并：分配新 id 并记录
            new_id = next_id
            next_id += 1
            self.itos[new_id] = self.itos[pair[0]] + self.itos[pair[1]]
            self.stoi[self.itos[new_id]] = new_id
            merges.append(pair)
            self.merges[pair] = new_id
            # 合并：只处理 pair_words 索引中的候选词（惰性校验，避免全量扫描）
            candidates = set(pair_words.get(key, ()))
            for wi in candidates:
                w = words[wi]
                idxs = [i for i in range(len(w) - 1) if w[i] * PK_BASE + w[i + 1] == key]
                if not idxs:
                    continue
                # 去掉该词全部旧相邻对计数
                for i in range(len(w) - 1):
                    global_count[w[i] * PK_BASE + w[i + 1]] -= 1
                # 生成新 word（array 紧凑存储，控内存）
                nw = array.array('I')
                i = 0
                L = len(w)
                while i < L:
                    if i + 1 < L and w[i] * PK_BASE + w[i + 1] == key:
                        nw.append(new_id)
                        i += 2
                    else:
                        nw.append(w[i])
                        i += 1
                # 新词相邻对计数 + 候选索引
                for i in range(len(nw) - 1):
                    k2 = nw[i] * PK_BASE + nw[i + 1]
                    global_count[k2] += 1
                    pair_words[k2].append(wi)
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
            ids = []
            for c in p:
                if c in self.stoi:
                    ids.append(self.stoi[c])
                elif self.byte_mode:
                    ids.extend(BYTE_OFFSET + b for b in _char_to_bytes(c))
                else:
                    ids.append(UNK)
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
        chars = []
        buf = []
        for i in ids:
            t = self.itos.get(i, '\ufffd')
            if self.byte_mode and t.startswith('<b'):
                buf.append(int(t[2:4], 16))
                continue
            if buf:
                chars.append(_bytes_to_char(buf))
                buf = []
            chars.append(t)
        if buf:
            chars.append(_bytes_to_char(buf))
        return ''.join(chars)

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
        self.byte_mode = any(k.startswith('<b') for k in self.itos.values())


if __name__ == '__main__':
    # 自测（byte 模式 + 单字保底）
    corpus = '这是一段中文测试文本。这是另一句测试。人工智能很厉害，机器学习是它的基础。'
    b = BPE()
    b.learn(corpus, vocab_target=40, extra_chars='靐龘', byte_mode=True)
    print('vocab', b.vocab_size, '| byte_mode', b.byte_mode)
    t = '这是靐龘另一句测试。'
    ids = b.encode(t)
    print('encoded', ids, '->', b.decode(ids))
    print('roundtrip ok:', b.decode(ids) == t)
    assert b.decode(b.encode(t)) == t, 'byte roundtrip 失败'
    # 旧模式自测（向后兼容）
    b2 = BPE()
    b2.learn(corpus, vocab_target=30, byte_mode=False)
    print('旧模式 roundtrip:', b2.decode(b2.encode('这是另一句测试。')) == '这是另一句测试。')
