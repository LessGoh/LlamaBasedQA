"""
Parallel Workflow Manager for ArXiv RAG System

This module implements the main ParallelWorkflowManager that orchestrates
the entire parallel document processing system. It coordinates the document
queue, workflow dispatcher, and progress tracking to provide a simple API
for batch document processing.

Based on the plan for creating a parallel document processing system with queue.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager

from .parallel_manager import (
    DocumentQueue,
    WorkflowDispatcher,
    ProgressTracker,
    DocumentTask,
    QueueStatus
)
from .workflow_events import (
    BatchProcessingStartEvent,
    BatchProcessingCompleteEvent,
    BatchProgressEvent,
    DocumentIndexedEvent,
    DocumentProcessingErrorEvent,
    ProcessingStatus,
    StageStatus
)
from .logging_config import log_batch_start, log_batch_complete



def _ensure_event_loop():
    """
    Ensure that we have a valid event loop for async operations
    
    Returns:
        The current or newly created event loop
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


@dataclass
class BatchProcessingConfig:
    """
    Configuration for batch processing operations
    """
    max_parallel_workflows: int = 2
    workflow_timeout: float = 1800.0  # 30 minutes
    queue_max_size: int = 20
    progress_update_interval: float = 5.0  # seconds
    graceful_shutdown_timeout: float = 300.0  # 5 minutes
    enable_retries: bool = True
    max_retries_per_document: int = 2


class BatchProcessingResult:
    """
    Container for batch processing results
    """
    
    def __init__(
        self,
        batch_id: str,
        total_documents: int,
        start_time: float,
        end_time: Optional[float] = None
    ):
        self.batch_id = batch_id
        self.total_documents = total_documents
        self.start_time = start_time
        self.end_time = end_time
        self.successful_results: List[DocumentIndexedEvent] = []
        self.failed_results: List[DocumentProcessingErrorEvent] = []
        self.processing_stats: Dict[str, Any] = {}
    
    @property
    def successful_count(self) -> int:
        return len(self.successful_results)
    
    @property
    def failed_count(self) -> int:
        return len(self.failed_results)
    
    @property
    def completion_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return (self.successful_count / self.total_documents) * 100.0
    
    @property
    def total_processing_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def is_completed(self) -> bool:
        return self.end_time is not None
    
    def add_result(self, result: Union[DocumentIndexedEvent, DocumentProcessingErrorEvent]):
        """Add a processing result"""
        if isinstance(result, DocumentIndexedEvent):
            self.successful_results.append(result)
        else:
            self.failed_results.append(result)
    
    def finalize(self, processing_stats: Dict[str, Any]):
        """Mark batch processing as complete"""
        self.end_time = time.time()
        self.processing_stats = processing_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "batch_id": self.batch_id,
            "total_documents": self.total_documents,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "completion_rate": self.completion_rate,
            "total_processing_time": self.total_processing_time,
            "is_completed": self.is_completed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "processing_stats": self.processing_stats
        }


class ParallelWorkflowManager:
    """
    Main coordinator for parallel document processing
    
    This class provides a simple API for batch document processing by
    coordinating the document queue, workflow dispatcher, and progress tracking.
    It handles error isolation, graceful shutdown, and real-time progress updates.
    """
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """
        Initialize the parallel workflow manager
        
        Args:
            config: Configuration for batch processing
        """
        # Ensure we have a valid event loop
        _ensure_event_loop()
        
        self.config = config or BatchProcessingConfig()
        
        # Initialize components
        self.document_queue = DocumentQueue(max_size=self.config.queue_max_size)
        self.workflow_dispatcher = WorkflowDispatcher(
            max_parallel_workflows=self.config.max_parallel_workflows,
            workflow_timeout=self.config.workflow_timeout
        )
        self.progress_tracker = ProgressTracker()
        
        # State management
        self.status = QueueStatus.IDLE
        self.current_batch: Optional[BatchProcessingResult] = None
        self.progress_callbacks: List[Callable[[BatchProgressEvent], None]] = []
        self.stop_event = asyncio.Event()
        self.processing_task: Optional[asyncio.Task] = None
        
    
    def _safe_async_run(self, coro):
        """
        Safely run async coroutine with event loop management
        
        Args:
            coro: Coroutine to execute
            
        Returns:
            Result of coroutine execution
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
            return loop.run_until_complete(coro)
        except (RuntimeError, AttributeError) as e:
            # Create new event loop if necessary
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                # Don't close the loop here as it might be needed for other operations
                pass
    
    async def start_batch_processing(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[BatchProgressEvent], None]] = None
    ) -> BatchProcessingResult:
        """
        Start batch processing of multiple documents
        
        Args:
            file_paths: List of PDF file paths to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchProcessingResult object for tracking progress
            
        Raises:
            RuntimeError: If system is already processing or invalid state
            ValueError: If file paths are invalid
        """
        if self.status != QueueStatus.IDLE:
            raise RuntimeError(f"Cannot start batch processing: system status is {self.status}")
        
        if not file_paths:
            raise ValueError("No file paths provided for batch processing")
        
        if len(file_paths) > self.config.queue_max_size:
            raise ValueError(f"Too many files: {len(file_paths)} > {self.config.queue_max_size}")
        
        
        # Generate batch ID and create result container
        batch_id = str(uuid.uuid4())
        self.current_batch = BatchProcessingResult(
            batch_id=batch_id,
            total_documents=len(file_paths),
            start_time=time.time()
        )
        
        # Log batch start
        config_info = {
            "max_parallel_workflows": self.config.max_parallel_workflows,
            "workflow_timeout": self.config.workflow_timeout
        }
        log_batch_start(len(file_paths), config_info)
        
        # Add progress callback if provided
        if progress_callback:
            self.progress_callbacks.append(progress_callback)
        
        try:
            # Add all files to queue
            tasks = []
            for i, file_path in enumerate(file_paths):
                task = await self.document_queue.add_task(file_path, priority=i)
                tasks.append(task)
                
                # Start progress tracking
                await self.progress_tracker.start_document_tracking(task)
            
            # Update status and start processing
            self.status = QueueStatus.RUNNING
            self.stop_event.clear()
            
            # Start the main processing loop
            self.processing_task = asyncio.create_task(
                self._process_queue_loop()
            )
            
            
            return self.current_batch
            
        except Exception as e:
            self.status = QueueStatus.ERROR
            raise
    
    async def stop_processing(self, wait_for_completion: bool = True) -> BatchProcessingResult:
        """
        Stop batch processing gracefully
        
        Args:
            wait_for_completion: Whether to wait for current tasks to complete
            
        Returns:
            Final BatchProcessingResult
        """
        if self.status not in [QueueStatus.RUNNING, QueueStatus.STOPPING]:
            return self.current_batch
        
        self.status = QueueStatus.STOPPING
        self.stop_event.set()
        
        if wait_for_completion and self.processing_task:
            try:
                # Wait for processing to complete with timeout
                await asyncio.wait_for(
                    self.processing_task,
                    timeout=self.config.graceful_shutdown_timeout
                )
            except asyncio.TimeoutError:
                self.processing_task.cancel()
        
        self.status = QueueStatus.STOPPED
        
        return self.current_batch
    
    async def get_current_progress(self) -> Optional[BatchProgressEvent]:
        """
        Get current batch processing progress
        
        Returns:
            BatchProgressEvent with current progress or None if not processing
        """
        if not self.current_batch:
            return None
        
        # Get statistics from queue and dispatcher
        queue_stats = await self.document_queue.get_stats()
        dispatcher_status = await self.workflow_dispatcher.get_status()
        progress_summary = await self.progress_tracker.get_progress_summary()
        
        # Calculate progress
        completed_count = queue_stats["completed_success"] + queue_stats["completed_error"]
        total_count = self.current_batch.total_documents
        
        # Estimate completion time
        if completed_count > 0 and self.current_batch.start_time:
            elapsed_time = time.time() - self.current_batch.start_time
            avg_time_per_doc = elapsed_time / completed_count
            remaining_docs = total_count - completed_count
            estimated_completion_time = remaining_docs * avg_time_per_doc if remaining_docs > 0 else 0
        else:
            estimated_completion_time = None
        
        return BatchProgressEvent(
            batch_id=self.current_batch.batch_id,
            completed_documents=completed_count,
            total_documents=total_count,
            active_workflows=dispatcher_status["busy_slots"],
            current_stage_info={
                "queue_stats": queue_stats,
                "dispatcher_status": dispatcher_status,
                "progress_summary": progress_summary
            },
            estimated_completion_time=estimated_completion_time,
            timestamp=time.time()
        )
    
    async def get_detailed_progress(self) -> Dict[str, Any]:
        """
        Get detailed progress information for all documents
        
        Returns:
            Detailed progress dictionary
        """
        if not self.current_batch:
            return {"error": "No active batch processing"}
        
        # Get all document progress
        all_progress = await self.progress_tracker.get_all_progress()
        
        # Get all tasks
        all_tasks = await self.document_queue.get_all_tasks()
        
        # Combine information
        detailed_progress = []
        for task in all_tasks:
            progress = next((p for p in all_progress if p.document_id == task.task_id), None)
            
            task_info = {
                "task_id": task.task_id,
                "filename": task.filename,
                "file_path": task.file_path,
                "status": task.status.value,
                "processing_time": task.processing_time,
                "error_count": task.error_count,
                "workflow_id": task.workflow_id,
                "stages": {}
            }
            
            if progress:
                task_info["stages"] = {
                    stage: status.value 
                    for stage, status in progress.stages_status.items()
                }
                task_info["current_stage"] = progress.current_stage
                task_info["stage_timings"] = progress.stage_timings
                task_info["error_info"] = progress.error_info
            
            detailed_progress.append(task_info)
        
        return {
            "batch_id": self.current_batch.batch_id,
            "status": self.status.value,
            "documents": detailed_progress,
            "summary": await self.get_current_progress()
        }
    
    async def _process_queue_loop(self):
        """
        Main processing loop that coordinates document processing
        """
        try:
            
            # Start progress update task
            progress_task = asyncio.create_task(self._progress_update_loop())
            
            # Create processing tasks
            processing_tasks = []
            for _ in range(self.config.max_parallel_workflows):
                task = asyncio.create_task(self._worker_loop())
                processing_tasks.append(task)
            
            # Wait for completion or stop signal
            await self._wait_for_completion_or_stop()
            
            # Cancel all tasks
            progress_task.cancel()
            for task in processing_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(progress_task, *processing_tasks, return_exceptions=True)
            
            # Finalize results
            await self._finalize_batch()
            
            
        except Exception as e:
            self.status = QueueStatus.ERROR
            raise
        finally:
            self.status = QueueStatus.STOPPED
    
    async def _worker_loop(self):
        """
        Worker loop that processes individual documents
        """
        try:
            while not self.stop_event.is_set():
                # Get next task from queue
                task = await self.document_queue.get_next_task()
                if not task:
                    # No more tasks, wait a bit before checking again
                    await asyncio.sleep(0.5)
                    continue
                
                try:
                    
                    # Execute the task
                    result = await self.workflow_dispatcher.execute_task(task)
                    
                    # Update progress tracking
                    if isinstance(result, DocumentIndexedEvent):
                        await self.progress_tracker.update_stage(
                            task.task_id, "indexing", StageStatus.COMPLETED
                        )
                    else:
                        await self.progress_tracker.mark_document_error(
                            task.task_id,
                            result.error_stage,
                            result.error_message,
                            result.error_type
                        )
                    
                    # Add result to batch
                    self.current_batch.add_result(result)
                    
                    # Mark task as completed
                    await self.document_queue.complete_task(task)
                    
                    
                except Exception as e:
                    
                    # Create error result
                    from .workflow_events import create_error_event
                    error_result = create_error_event(
                        document_id=task.task_id,
                        file_path=task.file_path,
                        filename=task.filename,
                        error_stage="worker",
                        error_message=f"Worker error: {str(e)}",
                        error_type=type(e).__name__,
                        timestamp=time.time()
                    )
                    
                    task.result = error_result
                    task.status = ProcessingStatus.ERROR
                    task.error_count += 1
                    
                    self.current_batch.add_result(error_result)
                    await self.document_queue.complete_task(task)
                    
        except asyncio.CancelledError:
            pass  # Handle cancellation gracefully
        except Exception as e:
            pass  # Handle worker errors gracefully
    
    async def _progress_update_loop(self):
        """
        Loop that sends periodic progress updates
        """
        try:
            while not self.stop_event.is_set():
                if self.progress_callbacks:
                    progress = await self.get_current_progress()
                    if progress:
                        for callback in self.progress_callbacks:
                            try:
                                callback(progress)
                            except Exception as e:
                                pass  # Handle progress callback errors gracefully
                
                await asyncio.sleep(self.config.progress_update_interval)
                
        except asyncio.CancelledError:
            pass  # Handle cancellation gracefully
        except Exception as e:
            pass  # Handle progress update errors gracefully
    
    async def _wait_for_completion_or_stop(self):
        """
        Wait for all documents to be processed or stop signal
        """
        while not self.stop_event.is_set():
            queue_stats = await self.document_queue.get_stats()
            
            # Check if all documents are processed
            if (queue_stats["pending"] == 0 and 
                queue_stats["processing"] == 0 and
                queue_stats["error"] == 0):
                break
            
            await asyncio.sleep(1.0)
    
    async def _finalize_batch(self):
        """
        Finalize the batch processing results
        """
        if not self.current_batch:
            return
        
        # Get final statistics
        queue_stats = await self.document_queue.get_stats()
        dispatcher_status = await self.workflow_dispatcher.get_status()
        progress_summary = await self.progress_tracker.get_progress_summary()
        
        processing_stats = {
            "queue_stats": queue_stats,
            "dispatcher_status": dispatcher_status,
            "progress_summary": progress_summary
        }
        
        # Finalize the batch
        self.current_batch.finalize(processing_stats)
        
        # Log batch completion
        log_batch_complete(
            self.current_batch.successful_count,
            self.current_batch.failed_count
        )
        
    
    def cleanup(self):
        """
        Clean up resources
        """
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
        
        self.workflow_dispatcher.cleanup()
        self.progress_callbacks.clear()
        
    
    @asynccontextmanager
    async def batch_processing_context(self, file_paths: List[str]):
        """
        Context manager for batch processing with automatic cleanup
        
        Usage:
            async with manager.batch_processing_context(file_paths) as result:
                # Monitor progress
                while not result.is_completed:
                    await asyncio.sleep(5)
                    progress = await manager.get_current_progress()
                    print(progress)
        """
        try:
            result = await self.start_batch_processing(file_paths)
            yield result
        finally:
            if self.status == QueueStatus.RUNNING:
                await self.stop_processing()
            self.cleanup()


# Convenience functions for easy usage
async def process_documents_batch(
    file_paths: List[str],
    config: Optional[BatchProcessingConfig] = None,
    progress_callback: Optional[Callable[[BatchProgressEvent], None]] = None
) -> BatchProcessingResult:
    """
    Process multiple documents in parallel with default settings
    
    Args:
        file_paths: List of PDF file paths to process
        config: Optional batch processing configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        BatchProcessingResult with processing outcomes
    """
    manager = ParallelWorkflowManager(config)
    
    try:
        async with manager.batch_processing_context(file_paths) as result:
            if progress_callback:
                manager.progress_callbacks.append(progress_callback)
            
            # Wait for completion
            while not result.is_completed:
                await asyncio.sleep(1.0)
            
            return result
    finally:
        manager.cleanup()


