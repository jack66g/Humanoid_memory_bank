# memory_store.py
import chromadb
from typing import Any
from .config import CogniConfig

class MemoryStore:
    def __init__(self, config: CogniConfig):
        """
        专职负责物理海马体（向量数据库）的连接、挂载与生命周期管理。
        完全解耦底层存储，向上层 Engine 提供原生的读写接口。
        """
        self.config = config
        
        # 原汁原味的持久化客户端初始化
        self.client = chromadb.PersistentClient(path=self.config.db_store_path)
        
        # 挂载长期记忆集合
        self.collection = self.client.get_or_create_collection(name=self.config.collection_name)

    def get_collection(self) -> Any:
        """
        将集合对象暴露给 CognitiveEngine。
        Engine 会直接调用这个对象的 add, query, update, delete 等原生方法。
        """
        return self.collection

    def wipe_all_memories(self):
        """
        【高危操作】物理学上的“脑白质切除术”。
        对应你原代码中 /clear 指令的强制抹除逻辑。
        """
        try:
            self.client.delete_collection(self.config.collection_name)
            self.collection = self.client.create_collection(self.config.collection_name)
            return True
        except Exception as e:
            print(f"⚠️ [警告] 记忆抹除失败: {e}")
            return False