"""
Advanced Progress Tracking and Monitoring System for ArXiv RAG System

This module provides detailed progress tracking, real-time monitoring,
and comprehensive reporting for the parallel document processing system.
It extends the basic ProgressTracker with advanced analytics and reporting.

Based on PHASE 3.3 of the parallel processing implementation plan.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union, Callable, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum

from .workflow_events import (
    DocumentStageProgress,
    ProcessingStatus,
    StageStatus,
    BatchProgressEvent,
    initialize_document_progress
)

# Set up logger
logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Types of progress reports"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    PERFORMANCE = "performance"
    ERROR_ANALYSIS = "error_analysis"
    TIMELINE = "timeline"


@dataclass
class StageMetrics:
    """Metrics for a single processing stage"""
    stage_name: str
    total_documents: int = 0
    completed_documents: int = 0
    failed_documents: int = 0
    total_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def avg_processing_time(self) -> float:
        return self.total_processing_time / max(self.completed_documents, 1)
    
    @property
    def success_rate(self) -> float:
        total = max(self.completed_documents + self.failed_documents, 1)
        return (self.completed_documents / total) * 100
    
    def update_timing(self, processing_time: float):
        """Update timing metrics"""
        self.total_processing_time += processing_time
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.max_processing_time = max(self.max_processing_time, processing_time)
    
    def add_error(self, error_type: str):
        """Record an error occurrence"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1


@dataclass 
class BatchMetrics:
    """Comprehensive metrics for a batch processing session"""
    batch_id: str
    start_time: float
    end_time: Optional[float] = None
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    stage_metrics: Dict[str, StageMetrics] = field(default_factory=dict)
    throughput_history: List[float] = field(default_factory=list)  # documents per minute
    resource_utilization: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    @property
    def completion_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return ((self.successful_documents + self.failed_documents) / self.total_documents) * 100
    
    @property
    def success_rate(self) -> float:
        total_processed = self.successful_documents + self.failed_documents
        if total_processed == 0:
            return 0.0
        return (self.successful_documents / total_processed) * 100
    
    @property
    def total_processing_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def avg_throughput(self) -> float:
        """Average documents processed per minute"""
        if not self.throughput_history:
            return 0.0
        return sum(self.throughput_history) / len(self.throughput_history)


@dataclass
class ProgressSnapshot:
    """Point-in-time snapshot of processing progress"""
    timestamp: float
    batch_id: str
    documents_completed: int
    documents_processing: int
    documents_pending: int
    documents_failed: int
    active_workflows: int
    avg_processing_time: float
    estimated_completion_time: Optional[float]
    throughput: float  # docs per minute
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdvancedProgressTracker:
    """
    Advanced progress tracking system with comprehensive monitoring capabilities
    
    Features:
    - Real-time progress monitoring
    - Performance analytics
    - Error analysis and reporting
    - Resource utilization tracking
    - Timeline analysis
    - Export capabilities
    """
    
    def __init__(self, snapshot_interval: float = 10.0, history_size: int = 1000):
        """
        Initialize advanced progress tracker
        
        Args:
            snapshot_interval: Interval for taking progress snapshots (seconds)
            history_size: Maximum number of snapshots to keep in memory
        """
        self.snapshot_interval = snapshot_interval
        self.history_size = history_size
        
        # Core tracking data
        self.document_progress: Dict[str, DocumentStageProgress] = {}
        self.batch_metrics: Dict[str, BatchMetrics] = {}
        self.progress_history: deque = deque(maxlen=history_size)
        
        # Monitoring state
        self.current_batch_id: Optional[str] = None
        self.monitoring_active = False
        self.snapshot_task: Optional[asyncio.Task] = None
        
        # Callbacks for real-time updates
        self.progress_callbacks: List[Callable[[ProgressSnapshot], None]] = []
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"AdvancedProgressTracker initialized (interval={snapshot_interval}s, history={history_size})")
    
    async def start_batch_monitoring(
        self,
        batch_id: str,
        total_documents: int,
        progress_callback: Optional[Callable[[ProgressSnapshot], None]] = None
    ):
        """
        Start monitoring a new batch processing session
        
        Args:
            batch_id: Unique batch identifier
            total_documents: Total number of documents in batch
            progress_callback: Optional callback for progress updates
        """
        async with self._lock:
            self.current_batch_id = batch_id
            
            # Initialize batch metrics
            self.batch_metrics[batch_id] = BatchMetrics(
                batch_id=batch_id,
                start_time=time.time(),
                total_documents=total_documents
            )
            
            # Add callback if provided
            if progress_callback:
                self.progress_callbacks.append(progress_callback)
            
            # Start monitoring task
            if not self.monitoring_active:
                self.monitoring_active = True
                self.snapshot_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"Started batch monitoring: {batch_id} ({total_documents} documents)")
    
    async def add_document_progress(self, progress: DocumentStageProgress):
        """Add or update document progress"""
        async with self._lock:
            self.document_progress[progress.document_id] = progress
    
    async def update_stage_completion(
        self,
        document_id: str,
        stage: str,
        status: StageStatus,
        processing_time: Optional[float] = None,
        error_type: Optional[str] = None
    ):
        """
        Update stage completion with detailed metrics tracking
        
        Args:
            document_id: Document identifier
            stage: Stage name
            status: Stage completion status
            processing_time: Time taken for this stage
            error_type: Error type if status is ERROR
        """
        async with self._lock:
            if document_id in self.document_progress:
                # Update document progress
                self.document_progress[document_id].update_stage_status(
                    stage, status, processing_time
                )
                
                # Update batch metrics
                if self.current_batch_id in self.batch_metrics:
                    batch = self.batch_metrics[self.current_batch_id]
                    
                    # Initialize stage metrics if needed
                    if stage not in batch.stage_metrics:
                        batch.stage_metrics[stage] = StageMetrics(stage_name=stage)
                    
                    stage_metrics = batch.stage_metrics[stage]
                    
                    if status == StageStatus.COMPLETED:
                        stage_metrics.completed_documents += 1
                        if processing_time:
                            stage_metrics.update_timing(processing_time)
                    elif status == StageStatus.ERROR:
                        stage_metrics.failed_documents += 1
                        if error_type:
                            stage_metrics.add_error(error_type)
    
    async def mark_document_complete(
        self,
        document_id: str,
        status: ProcessingStatus,
        total_processing_time: Optional[float] = None
    ):
        """Mark entire document processing as complete"""
        async with self._lock:
            if self.current_batch_id in self.batch_metrics:
                batch = self.batch_metrics[self.current_batch_id]
                
                if status == ProcessingStatus.SUCCESS:
                    batch.successful_documents += 1
                else:
                    batch.failed_documents += 1
    
    async def get_current_snapshot(self) -> Optional[ProgressSnapshot]:
        """Get current progress snapshot"""
        if not self.current_batch_id:
            return None
        
        async with self._lock:
            batch = self.batch_metrics.get(self.current_batch_id)
            if not batch:
                return None
            
            # Calculate current state
            documents_completed = batch.successful_documents + batch.failed_documents
            documents_processing = sum(
                1 for p in self.document_progress.values()
                if p.is_processing
            )
            documents_pending = batch.total_documents - documents_completed - documents_processing
            
            # Calculate metrics
            elapsed_time = time.time() - batch.start_time
            throughput = (documents_completed / max(elapsed_time / 60, 0.1))  # docs per minute
            
            # Estimate completion time
            if documents_completed > 0 and documents_pending > 0:
                avg_time_per_doc = elapsed_time / documents_completed
                estimated_completion_time = documents_pending * avg_time_per_doc
            else:
                estimated_completion_time = None
            
            # Calculate average processing time
            all_times = []
            for progress in self.document_progress.values():
                for timing in progress.stage_timings.values():
                    if timing > 0:
                        all_times.append(timing)
            
            avg_processing_time = sum(all_times) / max(len(all_times), 1)
            
            return ProgressSnapshot(
                timestamp=time.time(),
                batch_id=batch.batch_id,
                documents_completed=documents_completed,
                documents_processing=documents_processing,
                documents_pending=documents_pending,
                documents_failed=batch.failed_documents,
                active_workflows=documents_processing,  # Approximation
                avg_processing_time=avg_processing_time,
                estimated_completion_time=estimated_completion_time,
                throughput=throughput
            )
    
    async def get_detailed_report(self, report_type: ReportType = ReportType.SUMMARY) -> Dict[str, Any]:
        """
        Generate detailed progress report
        
        Args:
            report_type: Type of report to generate
            
        Returns:
            Comprehensive report dictionary
        """
        async with self._lock:
            if not self.current_batch_id:
                return {"error": "No active batch"}
            
            batch = self.batch_metrics.get(self.current_batch_id)
            if not batch:
                return {"error": "Batch metrics not found"}
            
            base_report = {
                "batch_id": batch.batch_id,
                "report_type": report_type.value,
                "generated_at": datetime.now().isoformat(),
                "batch_metrics": {
                    "total_documents": batch.total_documents,
                    "successful_documents": batch.successful_documents,
                    "failed_documents": batch.failed_documents,
                    "completion_rate": batch.completion_rate,
                    "success_rate": batch.success_rate,
                    "total_processing_time": batch.total_processing_time,
                    "avg_throughput": batch.avg_throughput
                }
            }
            
            if report_type == ReportType.DETAILED:
                base_report.update(await self._generate_detailed_report(batch))
            elif report_type == ReportType.PERFORMANCE:
                base_report.update(await self._generate_performance_report(batch))
            elif report_type == ReportType.ERROR_ANALYSIS:
                base_report.update(await self._generate_error_analysis_report(batch))
            elif report_type == ReportType.TIMELINE:
                base_report.update(await self._generate_timeline_report(batch))
            
            return base_report
    
    async def _generate_detailed_report(self, batch: BatchMetrics) -> Dict[str, Any]:
        """Generate detailed report section"""
        stage_details = {}
        for stage_name, stage_metrics in batch.stage_metrics.items():
            stage_details[stage_name] = {
                "total_documents": stage_metrics.total_documents,
                "completed_documents": stage_metrics.completed_documents,
                "failed_documents": stage_metrics.failed_documents,
                "success_rate": stage_metrics.success_rate,
                "avg_processing_time": stage_metrics.avg_processing_time,
                "min_processing_time": stage_metrics.min_processing_time,
                "max_processing_time": stage_metrics.max_processing_time,
                "error_counts": dict(stage_metrics.error_counts)
            }
        
        document_details = []
        for doc_id, progress in self.document_progress.items():
            doc_details = {
                "document_id": doc_id,
                "filename": progress.filename,
                "status": progress.status.value,
                "is_completed": progress.is_completed,
                "is_failed": progress.is_failed,
                "current_stage": progress.current_stage,
                "stages": {
                    stage: status.value
                    for stage, status in progress.stages_status.items()
                },
                "stage_timings": progress.stage_timings,
                "total_processing_time": progress.total_processing_time
            }
            
            if progress.error_info:
                doc_details["error_info"] = progress.error_info
            
            document_details.append(doc_details)
        
        return {
            "stage_details": stage_details,
            "document_details": document_details
        }
    
    async def _generate_performance_report(self, batch: BatchMetrics) -> Dict[str, Any]:
        """Generate performance analysis report"""
        # Calculate performance metrics
        processing_times = []
        stage_performance = defaultdict(list)
        
        for progress in self.document_progress.values():
            if progress.total_processing_time > 0:
                processing_times.append(progress.total_processing_time)
            
            for stage, timing in progress.stage_timings.items():
                if timing > 0:
                    stage_performance[stage].append(timing)
        
        # Calculate statistics
        performance_stats = {}
        if processing_times:
            processing_times.sort()
            n = len(processing_times)
            performance_stats = {
                "total_processing_time": {
                    "count": n,
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "mean": sum(processing_times) / n,
                    "median": processing_times[n // 2],
                    "p95": processing_times[int(n * 0.95)] if n >= 20 else processing_times[-1]
                }
            }
        
        stage_stats = {}
        for stage, times in stage_performance.items():
            if times:
                times.sort()
                n = len(times)
                stage_stats[stage] = {
                    "count": n,
                    "min": min(times),
                    "max": max(times),
                    "mean": sum(times) / n,
                    "median": times[n // 2]
                }
        
        return {
            "performance_statistics": performance_stats,
            "stage_statistics": stage_stats,
            "throughput_history": batch.throughput_history,
            "resource_utilization": dict(batch.resource_utilization)
        }
    
    async def _generate_error_analysis_report(self, batch: BatchMetrics) -> Dict[str, Any]:
        """Generate error analysis report"""
        error_summary = defaultdict(int)
        error_by_stage = defaultdict(lambda: defaultdict(int))
        failed_documents = []
        
        for progress in self.document_progress.values():
            if progress.is_failed and progress.error_info:
                error_info = progress.error_info
                error_type = error_info.get("error_type", "Unknown")
                stage = error_info.get("stage", "Unknown")
                
                error_summary[error_type] += 1
                error_by_stage[stage][error_type] += 1
                
                failed_documents.append({
                    "document_id": progress.document_id,
                    "filename": progress.filename,
                    "error_stage": stage,
                    "error_type": error_type,
                    "error_message": error_info.get("message", ""),
                    "processing_time": progress.total_processing_time
                })
        
        return {
            "error_summary": dict(error_summary),
            "errors_by_stage": {
                stage: dict(errors) for stage, errors in error_by_stage.items()
            },
            "failed_documents": failed_documents,
            "failure_patterns": await self._analyze_failure_patterns()
        }
    
    async def _generate_timeline_report(self, batch: BatchMetrics) -> Dict[str, Any]:
        """Generate timeline analysis report"""
        timeline_events = []
        
        # Create timeline from progress history
        for snapshot in self.progress_history:
            if isinstance(snapshot, ProgressSnapshot) and snapshot.batch_id == batch.batch_id:
                timeline_events.append({
                    "timestamp": snapshot.timestamp,
                    "datetime": datetime.fromtimestamp(snapshot.timestamp).isoformat(),
                    "documents_completed": snapshot.documents_completed,
                    "documents_processing": snapshot.documents_processing,
                    "throughput": snapshot.throughput,
                    "active_workflows": snapshot.active_workflows
                })
        
        return {
            "timeline_events": timeline_events,
            "processing_phases": await self._identify_processing_phases(timeline_events)
        }
    
    async def _analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze common failure patterns"""
        patterns = {
            "common_error_sequences": [],
            "problematic_file_types": [],
            "time_based_failures": [],
            "recommendations": []
        }
        
        # This could be expanded with more sophisticated analysis
        return patterns
    
    async def _identify_processing_phases(self, timeline_events: List[Dict]) -> List[Dict[str, Any]]:
        """Identify distinct phases in processing timeline"""
        phases = []
        
        if len(timeline_events) < 2:
            return phases
        
        # Simple phase detection based on throughput changes
        current_phase = {
            "start_time": timeline_events[0]["timestamp"],
            "phase_type": "startup",
            "avg_throughput": 0.0
        }
        
        throughputs = [event["throughput"] for event in timeline_events]
        avg_throughput = sum(throughputs) / len(throughputs)
        
        current_phase["end_time"] = timeline_events[-1]["timestamp"]
        current_phase["avg_throughput"] = avg_throughput
        phases.append(current_phase)
        
        return phases
    
    async def export_report(
        self,
        report_type: ReportType = ReportType.SUMMARY,
        format_type: str = "json",
        file_path: Optional[str] = None
    ) -> str:
        """
        Export progress report to file
        
        Args:
            report_type: Type of report to export
            format_type: Export format (json, csv)
            file_path: Optional file path, auto-generated if None
            
        Returns:
            Path to exported file
        """
        report = await self.get_detailed_report(report_type)
        
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = self.current_batch_id or "unknown"
            file_path = f"progress_report_{batch_id}_{report_type.value}_{timestamp}.{format_type}"
        
        if format_type.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Progress report exported to: {file_path}")
        return file_path
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        try:
            while self.monitoring_active:
                snapshot = await self.get_current_snapshot()
                if snapshot:
                    # Add to history
                    self.progress_history.append(snapshot)
                    
                    # Update batch throughput
                    if self.current_batch_id in self.batch_metrics:
                        self.batch_metrics[self.current_batch_id].throughput_history.append(
                            snapshot.throughput
                        )
                    
                    # Notify callbacks
                    for callback in self.progress_callbacks:
                        try:
                            callback(snapshot)
                        except Exception as e:
                            logger.error(f"Error in progress callback: {str(e)}")
                
                await asyncio.sleep(self.snapshot_interval)
        
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        
        if self.snapshot_task and not self.snapshot_task.done():
            self.snapshot_task.cancel()
            try:
                await self.snapshot_task
            except asyncio.CancelledError:
                pass
        
        # Finalize current batch
        if self.current_batch_id in self.batch_metrics:
            self.batch_metrics[self.current_batch_id].end_time = time.time()
        
        logger.info("Progress monitoring stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.progress_callbacks.clear()
        if self.monitoring_active:
            asyncio.create_task(self.stop_monitoring())
        
        logger.info("AdvancedProgressTracker cleaned up")


# Integration functions for backward compatibility
async def create_progress_tracker(
    snapshot_interval: float = 10.0,
    history_size: int = 1000
) -> AdvancedProgressTracker:
    """Create and initialize an advanced progress tracker"""
    return AdvancedProgressTracker(
        snapshot_interval=snapshot_interval,
        history_size=history_size
    )


def get_default_progress_tracker() -> AdvancedProgressTracker:
    """Get default progress tracker instance"""
    return AdvancedProgressTracker()


logger.info("Advanced progress tracker module initialized successfully")