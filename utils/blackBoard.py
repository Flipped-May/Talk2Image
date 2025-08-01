from dataclasses import dataclass, asdict
from typing import Dict, List, Callable, Optional
import threading
import json
from pathlib import Path

@dataclass(frozen=True)  # 不可变记录，确保可追溯性
class BlackboardRecord:
    """黑板中的单轮记录结构，对应论文中的$\mathcal{R}_t$"""
    user_input: str                  # 用户原始输入
    parsed_instruction: Dict        # 解析后的结构化指令（$I_t$）
    image_id: Optional[str] = None   # 图像唯一标识（路径或ID）
    judge_score: Optional[float] = None  # 评估分数
    clarification: Optional[str] = None  # 澄清提示（若有）
    turn: int = 0                    # 对话轮次

class Blackboard:
    """共享黑板通信中枢"""
    def __init__(self):
        self.records: List[BlackboardRecord] = []  # 所有记录
        self.lock = threading.RLock()              # 线程安全锁
        self.observers: Dict[str, List[Callable]] = {}  # 事件订阅者
        self.turn_counter = 0                      # 轮次计数器

    def write(self, record: BlackboardRecord) -> int:
        """写入记录并触发订阅事件"""
        with self.lock:
            # 自动填充轮次
            record = BlackboardRecord(
                user_input=record.user_input,
                parsed_instruction=record.parsed_instruction,
                image_id=record.image_id,
                judge_score=record.judge_score,
                clarification=record.clarification,
                turn=self.turn_counter
            )
            self.records.append(record)
            self.turn_counter += 1

            # 触发事件（如"生成完成"、"评估分数低"）
            self._trigger_events(record)
            return len(self.records) - 1  # 返回记录ID

    def read_latest(self) -> Optional[BlackboardRecord]:
        """读取最新记录"""
        with self.lock:
            return self.records[-1] if self.records else None

    def read_by_turn(self, turn: int) -> Optional[BlackboardRecord]:
        """按轮次读取记录"""
        with self.lock:
            return next((r for r in self.records if r.turn == turn), None)

    def subscribe(self, event_key: str, callback: Callable) -> None:
        """订阅事件（如"score_below_threshold"）"""
        with self.lock:
            if event_key not in self.observers:
                self.observers[event_key] = []
            self.observers[event_key].append(callback)

    def _trigger_events(self, record: BlackboardRecord) -> None:
        """根据记录内容触发匹配的事件"""
        # 1. 基础事件：轮次更新
        self._call_observers("turn_updated", record)
        
        # 2. 生成/编辑完成事件
        if record.image_id:
            self._call_observers("image_updated", record)
        
        # 3. 低分数事件（触发澄清）
        if record.judge_score is not None and record.judge_score < 0.4:
            self._call_observers("score_below_threshold", record)

    def _call_observers(self, event_key: str, record: BlackboardRecord) -> None:
        """调用事件订阅者"""
        with self.lock:
            if event_key in self.observers:
                for callback in self.observers[event_key]:
                    try:
                        callback(record)  # 异步执行回调（实际可加线程池）
                    except Exception as e:
                        print(f"黑板事件回调失败: {e}")