# engine.py
import time
import math
import uuid
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Callable, Tuple

from .config import CogniConfig

class CognitiveEngine:
    def __init__(
        self, 
        config: CogniConfig,
        db_collection: Any,
        get_emb_fn: Callable[[str], List[float]],
        eval_csn_fn: Callable[[str], Tuple[float, float, float]],
        calc_m_fn: Callable[[List[float], List[List[float]]], float],
        calc_v_fn: Callable[[float, float, float, float], float]
    ):
        """
        核心调度引擎：纯粹的业务中枢。
        绝不包含具体的模型挂载逻辑，只负责记忆的生老病死流转。
        """
        self.config = config
        self.long_term_db = db_collection
        
        # 依赖倒置：外部注入的能力（大模型打分、向量提取、纯物理公式计算）
        self._get_emb = get_emb_fn
        self.evaluate_csn_with_llm = eval_csn_fn
        self.calculate_physical_m = calc_m_fn
        self.calculate_v_score = calc_v_fn
        
        self.pending_queue = []

    # ==========================================
    # 【改动 1】：接收双轨数据（分离用户文本和AI文本）
    # ==========================================
    def add_to_pending(self, user_text: str, ai_text: str):
        memory_item = {
            "id": str(uuid.uuid4()),
            "user_text": user_text,  # 作为纯净的高维索引 (Key)
            "ai_text": ai_text,      # 作为上下文负荷 (Payload/Value)
            "timestamp": time.time()
        }
        self.pending_queue.append(memory_item)

    def process_pending_queue(self):
        """深夜模式：打分、张量共鸣计算、物理代谢"""
        if not self.pending_queue: 
            return
            
        print("\n🌙 [进入休眠状态] 开始清洗与物理张量比对...")
        survivors = 0
        
        while self.pending_queue:
            item = self.pending_queue.pop(0)
            
            # ==========================================
            # 【改动 2】：解包双轨数据，只拿用户的文本去计算！
            # ==========================================
            user_text = item["user_text"]
            ai_text = item["ai_text"]
            
            # 1. 第一步获取高质量 Mean Pooling 向量 (保证高维空间绝对纯净)
            current_vec = self._get_emb(user_text)
            
            # 2. 物理共鸣计算 (M) 与关联记忆拉取
            M = 5.0
            old_id = None
            old_text = None
            old_meta = None

            # 在循环内实时检查数据库条数，确保物理拦截即时生效
            if self.long_term_db.count() > 0:
                results = self.long_term_db.query(
                    query_embeddings=[current_vec], 
                    n_results=3, 
                    include=["embeddings", "documents", "metadatas"]
                )
                
                # 用 len() 代替隐式真值判断，避开数组多维歧义
                if results.get('embeddings') is not None and len(results['embeddings']) > 0 and len(results['embeddings'][0]) > 0:
                    old_vecs = results['embeddings'][0]
                    M = self.calculate_physical_m(current_vec, old_vecs)
                    
                    # 找出具体是哪一条记忆产生了最高共鸣 (原汁原味的张量运算)
                    t_new = torch.tensor(current_vec)
                    t_olds = torch.tensor(old_vecs)
                    sims = F.cosine_similarity(t_new.unsqueeze(0), t_olds)
                    max_idx = sims.argmax().item()
                    
                    old_id = results['ids'][0][max_idx]
                    old_text = results['documents'][0][max_idx]
                    old_meta = results['metadatas'][0][max_idx]

            # ==========================================
            # 【改动 3】：融合/覆盖机制，插入逻辑裁判员 2.0
            # ==========================================
            if M > self.config.m_consolidation_threshold and old_id is not None:
                
                # 调用裁判员判断是否冲突/改口 (接收的是字典返回值)
                judgment = self.evaluate_csn_with_llm.__self__.detect_and_correct_conflict(old_text, user_text)
                
                if judgment.get("is_conflict"):
                    # 💥 分支 A：发现歧义！精准覆盖记忆
                    corrected_fact = judgment.get("corrected_text")
                    print(f"  ⚡ [认知纠偏] 触发记忆重写！\n  旧：{old_text[:20]}...\n  新：{corrected_fact[:20]}...")
                    
                    # 生成修正后事实的新向量
                    new_fused_vec = self._get_emb(corrected_fact)
                    
                    # 覆写元数据
                    old_meta["timestamp"] = item["timestamp"]
                    old_meta["is_deprecated"] = False # 确保处于激活状态
                    
                    # 直接在原 ID 上覆写提纯后的文本和向量
                    self.long_term_db.update(
                        ids=[old_id], 
                        documents=[corrected_fact], 
                        embeddings=[new_fused_vec],
                        metadatas=[old_meta]
                    )
                    
                    # 修正完毕，直接跳出本次处理，不再将原始上下文作为新记忆入库
                    continue
                else:
                    # 🤝 分支 B：没有冲突，走原汁原味的【融合巩固逻辑】
                    old_meta["timestamp"] = item["timestamp"]
                    old_meta["initial_v"] = min(1.0, old_meta["initial_v"] + 0.1)
                    old_meta["ai_text"] = ai_text 
                    
                    if M > self.config.m_extreme_repeat_threshold: 
                        # 极度重复：仅更新老记忆元数据执行巩固，丢弃新文本
                        self.long_term_db.update(ids=[old_id], metadatas=[old_meta])
                        print(f"  🚫 [重复且巩固] 发现极度重复记忆 (M:{M:.1f})，老记忆已强化，AI回复已更新 | {user_text[:30]}...")
                    else:
                        # 相似融合：只合并用户的文本！并更新元数据执行巩固
                        fused_text = f"{old_text}；补充细节：{user_text}"
                        fused_vec = self._get_emb(fused_text)
                        self.long_term_db.update(
                            ids=[old_id],
                            documents=[fused_text],
                            embeddings=[fused_vec],
                            metadatas=[old_meta]
                        )
                        print(f"  🔄 [相似并融合巩固] 发现相似记忆 (M:{M:.1f})，老记忆已强化并融合细节 | {fused_text[:30]}...")
                    
                    # 融合完毕，跳出本次处理，进入下一条 pending
                    continue
            # ==========================================

            # 3. 让 LLM 客观打分 (C, S, N) (只评估用户的话)
            C, S, N = self.evaluate_csn_with_llm(user_text)
            
            # 4. 数学坍缩 (调用外部注入的公式)
            v_score = self.calculate_v_score(C, S, N, M)
            display_text = user_text.replace('\n', ' ')[:30] + "..."
            
            if v_score >= self.config.v_threshold:
                # ==========================================
                # 【改动 4】：作为新星系入库时，AI 的话隐入 Metadata
                # ==========================================
                self.long_term_db.add(
                    documents=[user_text], # 明面（用于检索）只存用户的文本
                    embeddings=[current_vec],
                    metadatas=[{
                        "timestamp": item["timestamp"], 
                        "initial_v": v_score,
                        "ai_text": ai_text  # AI的话被静静地封存进负荷中
                    }],
                    ids=[item["id"]]
                )
                survivors += 1
                bias_tag = " [张量共鸣]" if M > self.config.m_consolidation_threshold else ""
                print(f"  ✅ [巩固入库] V:{v_score:.2f} (C:{C},S:{S},N:{N},M:{M:.1f}){bias_tag} | {display_text}")
            else:
                print(f"  🗑️ [遗忘丢弃] V:{v_score:.2f} (C:{C},S:{S},N:{N},M:{M:.1f}) | {display_text}")
                
        print(f"🏁 [休眠结束] 本轮处理完成。")
        self.garbage_collect()

    def garbage_collect(self):
        """物理代谢：抹除衰减殆尽的枯萎记忆"""
        # 【逻辑原封不动】
        docs = self.long_term_db.get()
        if not docs or not docs['ids']: return
        
        now = time.time()
        to_delete_ids = []
        
        for i, meta in enumerate(docs['metadatas']):
            dt_hours = (now - meta['timestamp']) / 3600.0
            decay = math.exp(-self.config.forget_rate * dt_hours)
            if meta['initial_v'] * decay < self.config.death_threshold:
                to_delete_ids.append(docs['ids'][i])
                
        if to_delete_ids:
            self.long_term_db.delete(ids=to_delete_ids)
            print(f"💀 [新陈代谢] 艾宾浩斯死神降临，物理删除了 {len(to_delete_ids)} 条枯萎神经元！")

    def retrieve_memory(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        # 【逻辑原封不动】包括时间衰减算法与相似度公式全保留
        if self.long_term_db.count() == 0: return []
            
        query_vector = self._get_emb(query_text)
        results = self.long_term_db.query(
            query_embeddings=[query_vector],
            n_results=min(top_k * 3, self.long_term_db.count()) 
        )
        
        if not results['documents'] or not results['documents'][0]: return []

        now = time.time()
        scored_memories = []
        
        for i in range(len(results['documents'][0])):
            metadata = results['metadatas'][0][i]
            
            # 【仅新增此两行安全过滤】：如果在观念修正时被打上了作废标签，则提取时直接无视
            if metadata.get("is_deprecated", False):
                continue
                
            doc = results['documents'][0][i]
            mem_id = results['ids'][0][i] 
            dist = results['distances'][0][i]
            
            sim_score = 1.0 / (1.0 + dist)
            v_initial = metadata["initial_v"]
            mem_time = metadata["timestamp"]
            
            dt_hours = (now - mem_time) / 3600.0
            decay_factor = math.exp(-self.config.forget_rate * dt_hours)
            
            final_weight = sim_score * v_initial * decay_factor
            scored_memories.append({
                "text": doc, 
                "final_weight": final_weight,
                "id": mem_id,
                "meta": metadata
            })
            
        scored_memories.sort(key=lambda x: x["final_weight"], reverse=True)
        top_memories = scored_memories[:top_k]

        if top_memories:
            update_ids = []
            update_metas = []
            for mem in top_memories:
                update_ids.append(mem["id"])
                new_meta = mem["meta"].copy()
                new_meta["timestamp"] = now 
                new_meta["initial_v"] = min(1.0, new_meta["initial_v"] + 0.05) 
                update_metas.append(new_meta)
            
            self.long_term_db.update(ids=update_ids, metadatas=update_metas)

        return [{"text": m["text"], "final_weight": m["final_weight"], "meta": m["meta"]} for m in top_memories]