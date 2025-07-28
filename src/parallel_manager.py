"""
Parallel Processing Manager for ArXiv RAG System

This module implements the parallel document processing system with queue management.
It coordinates multiple DocumentProcessingWorkflow instances to process documents
concurrently while maintaining resource control and error isolation.

Based on the plan for creating a parallel document processing system with queue.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from .workflow_events import (
    BatchProcessingStartEvent,
    BatchProcessingCompleteEvent,
    BatchProgressEvent,
    DocumentIndexedEvent,
    DocumentProcessingErrorEvent,
    DocumentStageProgress,
    ProcessingStatus,
    StageStatus,
    initialize_document_progress
)
from .document_workflow import DocumentProcessingWorkflow



class QueueStatus(str, Enum):
    """Queue processing status"""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DocumentTask:
    """
    Represents a single document processing task in the queue
    """
    task_id: str
    file_path: str
    filename: str
    file_size_mb: float
    priority: int = 0  # Higher number = higher priority
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_count: int = 0
    max_retries: int = 2
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[Union[DocumentIndexedEvent, DocumentProcessingErrorEvent]] = None
    workflow_id: Optional[str] = None
    
    @property
    def is_retryable(self) -> bool:
        """Check if this task can be retried"""
        return self.error_count < self.max_retries and self.status == ProcessingStatus.ERROR
    
    @property
    def processing_time(self) -> float:
        """Calculate processing time if completed"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return 0.0


@dataclass
class WorkflowSlot:
    """
    Represents a workflow execution slot
    """
    slot_id: str
    workflow: DocumentProcessingWorkflow
    current_task: Optional[DocumentTask] = None
    is_busy: bool = False
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    def assign_task(self, task: DocumentTask):
        """Assign a task to this slot"""
        self.current_task = task
        self.is_busy = True
        self.last_activity = time.time()
        task.workflow_id = self.slot_id
        task.started_at = time.time()
        # Status already set to PROCESSING in get_next_task()
    
    def complete_task(self, result: Union[DocumentIndexedEvent, DocumentProcessingErrorEvent]):
        """Mark task as completed"""
        if self.current_task:
            self.current_task.completed_at = time.time()
            self.current_task.result = result
            if isinstance(result, DocumentProcessingErrorEvent):
                self.current_task.status = ProcessingStatus.ERROR
                self.current_task.error_count += 1
            else:
                self.current_task.status = ProcessingStatus.SUCCESS
        
        self.current_task = None
        self.is_busy = False
        self.last_activity = time.time()
    
    @property
    def is_available(self) -> bool:
        """Check if this slot is available for new tasks"""
        return not self.is_busy and self.current_task is None


class DocumentQueue:
    """
    Queue management system for document processing tasks
    
    Manages a priority queue of document processing tasks with support for
    retries, prioritization, and status tracking.
    """
    
    def __init__(self, max_size: int = 20):
        """
        Initialize the document queue
        
        Args:
            max_size: Maximum number of documents that can be queued
        """
        self.max_size = max_size
        self._queue: List[DocumentTask] = []
        self._completed: List[DocumentTask] = []
        self._lock = asyncio.Lock()
        
    
    async def add_task(self, file_path: str, priority: int = 0) -> DocumentTask:
        """
        Add a new document processing task to the queue
        
        Args:
            file_path: Path to the PDF file
            priority: Task priority (higher = processed first)
            
        Returns:
            DocumentTask object
            
        Raises:
            ValueError: If queue is full or file is invalid
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                raise ValueError(f"Queue is full (max {self.max_size} documents)")
            
            # Validate file
            path = Path(file_path)
            if not path.exists():
                raise ValueError(f"File not found: {file_path}")
            
            if not path.suffix.lower() == '.pdf':
                raise ValueError(f"Only PDF files are supported: {file_path}")
            
            # Calculate file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            
            # Create task
            task = DocumentTask(
                task_id=str(uuid.uuid4()),
                file_path=str(path.absolute()),
                filename=path.name,
                file_size_mb=file_size_mb,
                priority=priority
            )
            
            # Insert task in priority order (higher priority first)
            inserted = False
            for i, existing_task in enumerate(self._queue):
                if priority > existing_task.priority:
                    self._queue.insert(i, task)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(task)
            
            return task
    
    async def get_next_task(self) -> Optional[DocumentTask]:
        """
        Get the next task from the queue for processing
        
        Returns:
            Next DocumentTask or None if queue is empty
        """
        async with self._lock:
            # First check for retryable tasks
            for task in self._queue:
                if task.is_retryable:
                    task.status = ProcessingStatus.PROCESSING
                    return task
            
            # Get next pending task and mark as processing atomically
            for i, task in enumerate(self._queue):
                if task.status == ProcessingStatus.PENDING:
                    task.status = ProcessingStatus.PROCESSING
                    return task
            
            return None
    
    async def complete_task(self, task: DocumentTask):
        """
        Mark a task as completed and move it to completed list
        
        Args:
            task: The completed DocumentTask
        """
        async with self._lock:
            if task in self._queue:
                self._queue.remove(task)
                self._completed.append(task)
                
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get current queue statistics
        
        Returns:
            Dictionary with queue statistics
        """
        async with self._lock:
            pending = sum(1 for t in self._queue if t.status == ProcessingStatus.PENDING)
            processing = sum(1 for t in self._queue if t.status == ProcessingStatus.PROCESSING)
            error = sum(1 for t in self._queue if t.status == ProcessingStatus.ERROR)
            
            completed_success = sum(1 for t in self._completed if t.status == ProcessingStatus.SUCCESS)
            completed_error = sum(1 for t in self._completed if t.status == ProcessingStatus.ERROR)
            
            return {
                "queue_size": len(self._queue),
                "completed_size": len(self._completed),
                "pending": pending,
                "processing": processing,
                "error": error,
                "completed_success": completed_success,
                "completed_error": completed_error,
                "total_documents": len(self._queue) + len(self._completed)
            }
    
    async def get_all_tasks(self) -> List[DocumentTask]:
        """Get all tasks (queued and completed)"""
        async with self._lock:
            return self._queue + self._completed
    
    async def clear(self):
        """Clear all tasks from queue and completed list"""
        async with self._lock:
            self._queue.clear()
            self._completed.clear()


class WorkflowDispatcher:
    """
    Dispatcher for managing multiple workflow execution slots
    
    This class manages a fixed number of workflow slots and coordinates
    the assignment of tasks from the queue to available workflows.
    """
    
    def __init__(self, max_parallel_workflows: int = 2, workflow_timeout: float = 1800.0):
        """
        Initialize the workflow dispatcher
        
        Args:
            max_parallel_workflows: Maximum number of parallel workflows
            workflow_timeout: Timeout for individual workflow execution
        """
        self.max_parallel_workflows = max_parallel_workflows
        self.workflow_timeout = workflow_timeout
        self.slots: Dict[str, WorkflowSlot] = {}
        self.semaphore = asyncio.Semaphore(max_parallel_workflows)
        self._running_tasks: Set[asyncio.Task] = set()
        
        # Initialize workflow slots
        for i in range(max_parallel_workflows):
            slot_id = f"workflow_slot_{i}"
            workflow = DocumentProcessingWorkflow(
                timeout=workflow_timeout,
                verbose=False
            )
            workflow.initialize_processors()
            
            self.slots[slot_id] = WorkflowSlot(
                slot_id=slot_id,
                workflow=workflow
            )
        
    
    def get_available_slot(self) -> Optional[WorkflowSlot]:
        """
        Get an available workflow slot
        
        Returns:
            Available WorkflowSlot or None if all are busy
        """
        for slot in self.slots.values():
            if slot.is_available:
                return slot
        return None
    
    async def execute_task(
        self,
        task: DocumentTask,
        progress_callback: Optional[callable] = None
    ) -> Union[DocumentIndexedEvent, DocumentProcessingErrorEvent]:
        """
        Execute a single document processing task
        
        Args:
            task: DocumentTask to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing result event
        """
        async with self.semaphore:
            slot = self.get_available_slot()
            if not slot:
                raise RuntimeError("No available workflow slots")
            
            try:
                slot.assign_task(task)
                
                # Execute the workflow
                result = await slot.workflow.run(file_path=task.file_path)
                
                # Handle result
                if hasattr(result, 'result'):
                    # Extract the actual result from StopEvent
                    actual_result = result.result if hasattr(result, 'result') else result
                else:
                    actual_result = result
                
                slot.complete_task(actual_result)
                
                return actual_result
                
            except Exception as e:
                error_msg = f"Workflow execution failed: {str(e)}"
                
                # Create error result
                from .workflow_events import create_error_event
                error_result = create_error_event(
                    document_id=task.task_id,
                    file_path=task.file_path,
                    filename=task.filename,
                    error_stage="workflow",
                    error_message=error_msg,
                    error_type=type(e).__name__,
                    timestamp=time.time()
                )
                
                slot.complete_task(error_result)
                return error_result
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current dispatcher status
        
        Returns:
            Dictionary with dispatcher status information
        """
        busy_slots = sum(1 for slot in self.slots.values() if slot.is_busy)
        available_slots = len(self.slots) - busy_slots
        
        slot_info = []
        for slot in self.slots.values():
            info = {
                "slot_id": slot.slot_id,
                "is_busy": slot.is_busy,
                "current_task": {
                    "task_id": slot.current_task.task_id,
                    "filename": slot.current_task.filename,
                    "processing_time": slot.current_task.processing_time
                } if slot.current_task else None,
                "last_activity": slot.last_activity
            }
            slot_info.append(info)
        
        return {
            "total_slots": len(self.slots),
            "busy_slots": busy_slots,
            "available_slots": available_slots,
            "running_tasks": len(self._running_tasks),
            "slot_details": slot_info
        }
    
    def cleanup(self):
        """Clean up resources"""
        # Cancel any running tasks
        for task in self._running_tasks:
            if not task.done():
                task.cancel()
        self._running_tasks.clear()
        


class ProgressTracker:
    """
    Tracks detailed progress of each document through processing stages
    """
    
    def __init__(self):
        """Initialize progress tracker"""
        self.document_progress: Dict[str, DocumentStageProgress] = {}
        self._lock = asyncio.Lock()
        
    
    async def start_document_tracking(self, task: DocumentTask):
        """
        Start tracking progress for a new document
        
        Args:
            task: DocumentTask to track
        """
        async with self._lock:
            progress = initialize_document_progress(
                document_id=task.task_id,
                filename=task.filename,
                file_path=task.file_path,
                start_time=task.created_at
            )
            self.document_progress[task.task_id] = progress
            
    
    async def update_stage(
        self,
        document_id: str,
        stage: str,
        status: StageStatus,
        timing: Optional[float] = None
    ):
        """
        Update the status of a processing stage
        
        Args:
            document_id: Document identifier
            stage: Stage name
            status: New stage status
            timing: Time taken for this stage (if completed)
        """
        async with self._lock:
            if document_id in self.document_progress:
                self.document_progress[document_id].update_stage_status(stage, status, timing)
    
    async def mark_document_error(
        self,
        document_id: str,
        stage: str,
        error_message: str,
        error_type: str
    ):
        """
        Mark a document as failed at a specific stage
        
        Args:
            document_id: Document identifier
            stage: Stage where error occurred
            error_message: Error description
            error_type: Type of error
        """
        async with self._lock:
            if document_id in self.document_progress:
                self.document_progress[document_id].mark_error(stage, error_message, error_type)
    
    async def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get summary of all document progress
        
        Returns:
            Dictionary with progress summary
        """
        async with self._lock:
            total_docs = len(self.document_progress)
            completed = sum(1 for p in self.document_progress.values() if p.is_completed)
            failed = sum(1 for p in self.document_progress.values() if p.is_failed)
            processing = total_docs - completed - failed
            
            return {
                "total_documents": total_docs,
                "completed": completed,
                "failed": failed,
                "processing": processing,
                "completion_rate": (completed / total_docs) * 100 if total_docs > 0 else 0,
                "failure_rate": (failed / total_docs) * 100 if total_docs > 0 else 0
            }
    
    async def get_document_progress(self, document_id: str) -> Optional[DocumentStageProgress]:
        """Get progress for a specific document"""
        async with self._lock:
            return self.document_progress.get(document_id)
    
    async def get_all_progress(self) -> List[DocumentStageProgress]:
        """Get progress for all documents"""
        async with self._lock:
            return list(self.document_progress.values())


