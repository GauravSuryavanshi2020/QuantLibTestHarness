import asyncio
import threading
import time
import random
import concurrent.futures
import psutil
import gc
import json
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Callable, Optional, Union
import logging
from contextlib import contextmanager
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from enum import Enum

# QuantLib imports (assumed to be available)
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("QuantLib not available - using mock implementation")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChaosType(Enum):
    """Types of chaos engineering scenarios"""
    NETWORK_DELAY = "network_delay"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

@dataclass
class TestMetrics:
    """Comprehensive metrics for QuantLib operations"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    peak_memory_usage: float = 0.0
    avg_memory_usage: float = 0.0
    cpu_usage_peak: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    pricing_errors: int = 0
    calculation_errors: int = 0
    convergence_failures: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    gc_collections: int = 0

@dataclass
class ChaosEvent:
    """Chaos engineering event configuration"""
    chaos_type: ChaosType
    probability: float
    duration: float
    intensity: float
    description: str

class QuantLibMock:
    """Mock QuantLib for testing when library is not available"""
    
    class Date:
        def __init__(self, day, month, year):
            self.day, self.month, self.year = day, month, year
    
    class BlackScholesMertonProcess:
        def __init__(self, *args): pass
    
    class EuropeanOption:
        def __init__(self, *args): pass
        def NPV(self): return random.uniform(0.5, 50.0)
        def delta(self): return random.uniform(0.1, 0.9)
        def gamma(self): return random.uniform(0.01, 0.1)
        def theta(self): return random.uniform(-0.5, 0.0)
        def vega(self): return random.uniform(0.1, 2.0)
    
    class AnalyticEuropeanEngine:
        def __init__(self, *args): pass
    
    class PlainVanillaPayoff:
        def __init__(self, *args): pass
    
    class EuropeanExercise:
        def __init__(self, *args): pass
    
    class SimpleQuote:
        def __init__(self, value): self.value = value
        def setValue(self, value): self.value = value
    
    class YieldTermStructureHandle:
        def __init__(self, *args): pass
    
    class BlackVolTermStructureHandle:
        def __init__(self, *args): pass
    
    class QuoteHandle:
        def __init__(self, *args): pass
    
    class FlatForward:
        def __init__(self, *args): pass
    
    class BlackConstantVol:
        def __init__(self, *args): pass
    
    # Constants
    Option = type('Option', (), {'Call': 1, 'Put': -1})
    Actual365Fixed = lambda: None
    TARGET = lambda: None
    Settings = type('Settings', (), {'instance': lambda: type('instance', (), {'evaluationDate': None})()})

# Use QuantLib if available, otherwise use mock
if QUANTLIB_AVAILABLE:
    ql_lib = ql
else:
    ql_lib = QuantLibMock()

class ResourceMonitor:
    """Monitor system resources during test execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent,
                    'gc_count': sum(gc.get_count())
                })
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB"""
        if not self.metrics:
            return 0.0
        return max(m['memory_mb'] for m in self.metrics)
    
    def get_avg_memory(self) -> float:
        """Get average memory usage in MB"""
        if not self.metrics:
            return 0.0
        return sum(m['memory_mb'] for m in self.metrics) / len(self.metrics)
    
    def get_peak_cpu(self) -> float:
        """Get peak CPU usage percentage"""
        if not self.metrics:
            return 0.0
        return max(m['cpu_percent'] for m in self.metrics)

class ChaosEngineer:
    """Chaos engineering for QuantLib testing"""
    
    def __init__(self):
        self.active_chaos = []
        self.chaos_events = [
            ChaosEvent(ChaosType.NETWORK_DELAY, 0.05, 0.5, 0.1, "Network delay simulation"),
            ChaosEvent(ChaosType.MEMORY_PRESSURE, 0.02, 2.0, 0.8, "Memory pressure simulation"),
            ChaosEvent(ChaosType.CPU_SPIKE, 0.03, 1.0, 0.9, "CPU spike simulation"),
            ChaosEvent(ChaosType.TIMEOUT, 0.01, 0.0, 1.0, "Operation timeout"),
            ChaosEvent(ChaosType.EXCEPTION, 0.02, 0.0, 1.0, "Random exception"),
        ]
    
    async def maybe_introduce_chaos(self):
        """Randomly introduce chaos events"""
        for event in self.chaos_events:
            if random.random() < event.probability:
                await self._execute_chaos_event(event)
    
    async def _execute_chaos_event(self, event: ChaosEvent):
        """Execute a specific chaos event"""
        logger.info(f"Introducing chaos: {event.description}")
        
        if event.chaos_type == ChaosType.NETWORK_DELAY:
            await asyncio.sleep(event.duration * event.intensity)
        
        elif event.chaos_type == ChaosType.MEMORY_PRESSURE:
            # Allocate memory to simulate pressure
            memory_hog = []
            try:
                for _ in range(int(1000 * event.intensity)):
                    memory_hog.append([0] * 10000)  # Allocate arrays
                await asyncio.sleep(event.duration)
            finally:
                del memory_hog
                gc.collect()
        
        elif event.chaos_type == ChaosType.CPU_SPIKE:
            # CPU intensive operation
            start_time = time.time()
            while time.time() - start_time < event.duration:
                _ = sum(i**2 for i in range(int(10000 * event.intensity)))
                await asyncio.sleep(0.001)  # Yield control
        
        elif event.chaos_type == ChaosType.TIMEOUT:
            if random.random() < event.intensity:
                raise asyncio.TimeoutError("Chaos-induced timeout")
        
        elif event.chaos_type == ChaosType.EXCEPTION:
            if random.random() < event.intensity:
                raise Exception("Chaos-induced exception")

class QuantLibTestScenario:
    """Base class for QuantLib-specific test scenarios"""
    
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def setup_market_data(self):
        """Setup common market data"""
        # Evaluation date
        today = ql_lib.Date(15, 1, 2024)
        if hasattr(ql_lib, 'Settings'):
            ql_lib.Settings.instance().evaluationDate = today
        
        # Market data
        spot = 100.0
        risk_free_rate = 0.05
        volatility = 0.20
        dividend_yield = 0.0
        
        # Create market objects
        spot_handle = ql_lib.QuoteHandle(ql_lib.SimpleQuote(spot))
        
        risk_free_ts = ql_lib.YieldTermStructureHandle(
            ql_lib.FlatForward(today, risk_free_rate, ql_lib.Actual365Fixed())
        )
        
        dividend_ts = ql_lib.YieldTermStructureHandle(
            ql_lib.FlatForward(today, dividend_yield, ql_lib.Actual365Fixed())
        )
        
        vol_ts = ql_lib.BlackVolTermStructureHandle(
            ql_lib.BlackConstantVol(today, ql_lib.TARGET(), volatility, ql_lib.Actual365Fixed())
        )
        
        return spot_handle, risk_free_ts, dividend_ts, vol_ts
    
    def create_european_option(self, strike: float, maturity_days: int = 30, option_type: str = "Call"):
        """Create a European option"""
        cache_key = f"option_{strike}_{maturity_days}_{option_type}"
        
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Setup market data
        spot_handle, risk_free_ts, dividend_ts, vol_ts = self.setup_market_data()
        
        # Option parameters
        maturity = ql_lib.Date(15 + maturity_days, 1, 2024)
        payoff = ql_lib.PlainVanillaPayoff(
            ql_lib.Option.Call if option_type == "Call" else ql_lib.Option.Put, 
            strike
        )
        exercise = ql_lib.EuropeanExercise(maturity)
        
        # Create option
        option = ql_lib.EuropeanOption(payoff, exercise)
        
        # Create pricing engine
        process = ql_lib.BlackScholesMertonProcess(
            spot_handle, dividend_ts, risk_free_ts, vol_ts
        )
        engine = ql_lib.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)
        
        # Cache the option
        self.cache[cache_key] = option
        
        return option

class QuantLibUsagePattern(ABC):
    """Base class for QuantLib-specific usage patterns"""
    
    def __init__(self):
        self.scenario = QuantLibTestScenario()
        self.chaos_engineer = ChaosEngineer()
        self.resource_monitor = ResourceMonitor()
    
    @abstractmethod
    async def execute(self, duration: int) -> TestMetrics:
        pass
    
    async def _price_option(self, strike: float = None, maturity: int = None, option_type: str = None) -> Dict[str, float]:
        """Price an option and return Greeks"""
        strike = strike or random.uniform(80, 120)
        maturity = maturity or random.randint(1, 365)
        option_type = option_type or random.choice(["Call", "Put"])
        
        # Introduce chaos
        await self.chaos_engineer.maybe_introduce_chaos()
        
        option = self.scenario.create_european_option(strike, maturity, option_type)
        
        # Calculate pricing metrics
        results = {
            'npv': option.NPV(),
            'delta': option.delta(),
            'gamma': option.gamma(),
            'theta': option.theta(),
            'vega': option.vega()
        }
        
        return results

class HighFrequencyTradingPattern(QuantLibUsagePattern):
    """Simulates high-frequency trading option pricing"""
    
    def __init__(self, requests_per_second: int = 100):
        super().__init__()
        self.requests_per_second = requests_per_second
    
    async def execute(self, duration: int) -> TestMetrics:
        metrics = TestMetrics()
        self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        interval = 1.0 / self.requests_per_second
        
        try:
            while time.time() - start_time < duration:
                operation_start = time.time()
                
                try:
                    # Rapid-fire option pricing
                    results = await self._price_option()
                    metrics.successful_operations += 1
                    
                    response_time = time.time() - operation_start
                    metrics.average_response_time = (
                        (metrics.average_response_time * (metrics.successful_operations - 1) + response_time) 
                        / metrics.successful_operations
                    )
                    metrics.min_response_time = min(metrics.min_response_time, response_time)
                    metrics.max_response_time = max(metrics.max_response_time, response_time)
                    
                except Exception as e:
                    metrics.failed_operations += 1
                    metrics.pricing_errors += 1
                    logger.error(f"HFT pricing failed: {e}")
                
                metrics.total_operations += 1
                
                # Maintain high frequency
                elapsed = time.time() - operation_start
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
        
        finally:
            self.resource_monitor.stop_monitoring()
        
        # Finalize metrics
        metrics.error_rate = metrics.failed_operations / max(metrics.total_operations, 1)
        metrics.throughput = metrics.total_operations / duration
        metrics.peak_memory_usage = self.resource_monitor.get_peak_memory()
        metrics.avg_memory_usage = self.resource_monitor.get_avg_memory()
        metrics.cpu_usage_peak = self.resource_monitor.get_peak_cpu()
        metrics.cache_hits = self.scenario.cache_hits
        metrics.cache_misses = self.scenario.cache_misses
        
        return metrics

class PortfolioRiskAnalysisPattern(QuantLibUsagePattern):
    """Simulates portfolio risk analysis with complex calculations"""
    
    def __init__(self, portfolio_size: int = 100):
        super().__init__()
        self.portfolio_size = portfolio_size
    
    async def execute(self, duration: int) -> TestMetrics:
        metrics = TestMetrics()
        self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                operation_start = time.time()
                
                try:
                    # Simulate portfolio analysis
                    portfolio_values = []
                    
                    # Price multiple options in portfolio
                    for i in range(self.portfolio_size):
                        strike = random.uniform(80, 120)
                        maturity = random.randint(1, 365)
                        option_type = random.choice(["Call", "Put"])
                        
                        results = await self._price_option(strike, maturity, option_type)
                        portfolio_values.append(results['npv'])
                    
                    # Calculate portfolio metrics
                    portfolio_npv = sum(portfolio_values)
                    portfolio_var = np.var(portfolio_values) if len(portfolio_values) > 1 else 0
                    
                    metrics.successful_operations += 1
                    
                    response_time = time.time() - operation_start
                    metrics.average_response_time = (
                        (metrics.average_response_time * (metrics.successful_operations - 1) + response_time) 
                        / metrics.successful_operations
                    )
                    
                except Exception as e:
                    metrics.failed_operations += 1
                    metrics.calculation_errors += 1
                    logger.error(f"Portfolio analysis failed: {e}")
                
                metrics.total_operations += 1
                
                # Periodic garbage collection
                if metrics.total_operations % 10 == 0:
                    gc.collect()
                    metrics.gc_collections += 1
        
        finally:
            self.resource_monitor.stop_monitoring()
        
        # Finalize metrics
        metrics.error_rate = metrics.failed_operations / max(metrics.total_operations, 1)
        metrics.throughput = metrics.total_operations / duration
        metrics.peak_memory_usage = self.resource_monitor.get_peak_memory()
        metrics.avg_memory_usage = self.resource_monitor.get_avg_memory()
        metrics.cpu_usage_peak = self.resource_monitor.get_peak_cpu()
        metrics.cache_hits = self.scenario.cache_hits
        metrics.cache_misses = self.scenario.cache_misses
        
        return metrics

class VolatilityCalibrationPattern(QuantLibUsagePattern):
    """Simulates volatility surface calibration"""
    
    def __init__(self, calibration_points: int = 50):
        super().__init__()
        self.calibration_points = calibration_points
    
    async def execute(self, duration: int) -> TestMetrics:
        metrics = TestMetrics()
        self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                operation_start = time.time()
                
                try:
                    # Simulate volatility calibration
                    calibration_errors = []
                    
                    for i in range(self.calibration_points):
                        # Different strikes and maturities for calibration
                        strike = 90 + i * 0.5
                        maturity = 30 + i * 10
                        
                        try:
                            results = await self._price_option(strike, maturity)
                            
                            # Simulate calibration error
                            market_price = results['npv'] * (1 + random.uniform(-0.05, 0.05))
                            model_price = results['npv']
                            error = abs(market_price - model_price) / market_price
                            
                            calibration_errors.append(error)
                            
                        except Exception as e:
                            metrics.convergence_failures += 1
                            logger.warning(f"Calibration point failed: {e}")
                    
                    # Calculate calibration quality
                    avg_error = np.mean(calibration_errors) if calibration_errors else 1.0
                    
                    if avg_error < 0.01:  # Good calibration
                        metrics.successful_operations += 1
                    else:
                        metrics.convergence_failures += 1
                    
                    response_time = time.time() - operation_start
                    metrics.average_response_time = (
                        (metrics.average_response_time * max(metrics.successful_operations, 1) + response_time) 
                        / (metrics.successful_operations + 1)
                    )
                    
                except Exception as e:
                    metrics.failed_operations += 1
                    metrics.calculation_errors += 1
                    logger.error(f"Volatility calibration failed: {e}")
                
                metrics.total_operations += 1
        
        finally:
            self.resource_monitor.stop_monitoring()
        
        # Finalize metrics
        metrics.error_rate = metrics.failed_operations / max(metrics.total_operations, 1)
        metrics.throughput = metrics.total_operations / duration
        metrics.peak_memory_usage = self.resource_monitor.get_peak_memory()
        metrics.avg_memory_usage = self.resource_monitor.get_avg_memory()
        metrics.cpu_usage_peak = self.resource_monitor.get_peak_cpu()
        metrics.cache_hits = self.scenario.cache_hits
        metrics.cache_misses = self.scenario.cache_misses
        
        return metrics

class MarketDataFeedPattern(QuantLibUsagePattern):
    """Simulates real-time market data processing"""
    
    def __init__(self, feed_rate: int = 10):
        super().__init__()
        self.feed_rate = feed_rate
        self.market_data_log = []
    
    async def execute(self, duration: int) -> TestMetrics:
        metrics = TestMetrics()
        self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                operation_start = time.time()
                
                try:
                    # Simulate market data update
                    new_spot = 100 * (1 + random.uniform(-0.02, 0.02))
                    new_vol = 0.20 * (1 + random.uniform(-0.1, 0.1))
                    
                    # Update pricing based on new market data
                    results = await self._price_option(strike=100, maturity=30)
                    
                    # Log market data
                    self.market_data_log.append({
                        'timestamp': time.time(),
                        'spot': new_spot,
                        'volatility': new_vol,
                        'option_price': results['npv']
                    })
                    
                    # Keep only recent data
                    if len(self.market_data_log) > 1000:
                        self.market_data_log = self.market_data_log[-500:]
                    
                    metrics.successful_operations += 1
                    
                    response_time = time.time() - operation_start
                    metrics.average_response_time = (
                        (metrics.average_response_time * (metrics.successful_operations - 1) + response_time) 
                        / metrics.successful_operations
                    )
                    
                except Exception as e:
                    metrics.failed_operations += 1
                    logger.error(f"Market data processing failed: {e}")
                
                metrics.total_operations += 1
                
                # Maintain feed rate
                await asyncio.sleep(1.0 / self.feed_rate)
        
        finally:
            self.resource_monitor.stop_monitoring()
        
        # Finalize metrics
        metrics.error_rate = metrics.failed_operations / max(metrics.total_operations, 1)
        metrics.throughput = metrics.total_operations / duration
        metrics.peak_memory_usage = self.resource_monitor.get_peak_memory()
        metrics.avg_memory_usage = self.resource_monitor.get_avg_memory()
        metrics.cpu_usage_peak = self.resource_monitor.get_peak_cpu()
        metrics.cache_hits = self.scenario.cache_hits
        metrics.cache_misses = self.scenario.cache_misses
        
        return metrics

class GradualLoadRamping:
    """Gradually increase load to find breaking points"""
    
    def __init__(self, pattern_class, start_load: int = 1, max_load: int = 100, ramp_duration: int = 300):
        self.pattern_class = pattern_class
        self.start_load = start_load
        self.max_load = max_load
        self.ramp_duration = ramp_duration
        self.breaking_point = None
    
    async def execute(self) -> Dict[str, Any]:
        """Execute gradual load ramping"""
        results = []
        current_load = self.start_load
        
        logger.info(f"Starting gradual load ramping from {self.start_load} to {self.max_load}")
        
        while current_load <= self.max_load:
            logger.info(f"Testing load level: {current_load}")
            
            # Create pattern with current load
            if self.pattern_class == HighFrequencyTradingPattern:
                pattern = self.pattern_class(requests_per_second=current_load)
            elif self.pattern_class == PortfolioRiskAnalysisPattern:
                pattern = self.pattern_class(portfolio_size=current_load)
            else:
                pattern = self.pattern_class()
            
            # Run test for shorter duration during ramping
            test_duration = min(30, self.ramp_duration // 10)
            
            try:
                metrics = await pattern.execute(test_duration)
                
                results.append({
                    'load_level': current_load,
                    'metrics': asdict(metrics),
                    'timestamp': time.time()
                })
                
                # Check for breaking point
                if metrics.error_rate > 0.05 or metrics.average_response_time > 5.0:
                    self.breaking_point = current_load
                    logger.warning(f"Breaking point detected at load level: {current_load}")
                    break
                
            except Exception as e:
                logger.error(f"Load test failed at level {current_load}: {e}")
                self.breaking_point = current_load
                break
            
            # Increase load
            current_load += max(1, current_load // 10)  # 10% increments
        
        return {
            'breaking_point': self.breaking_point,
            'max_tested_load': current_load,
            'results': results
        }

class DataDrivenScenarios:
    """Execute scenarios based on production log data"""
    
    def __init__(self, log_file_path: str = None):
        self.log_file_path = log_file_path
        self.scenarios = []
    
    def load_production_logs(self, log_data: List[Dict] = None) -> None:
        """Load production log data or use sample data"""
        if log_data:
            self.scenarios = log_data
        else:
            # Generate sample production-like scenarios
            self.scenarios = [
                {
                    'timestamp': time.time() - i * 3600,
                    'operation_type': random.choice(['option_pricing', 'portfolio_analysis', 'volatility_calibration']),
                    'parameters': {
                        'strike': random.uniform(80, 120),
                        'maturity': random.randint(1, 365),
                        'option_type': random.choice(['Call', 'Put']),
                        'portfolio_size': random.randint(10, 200)
                    },
                    'expected_duration': random.uniform(0.01, 2.0)
                }
                for i in range(1000)
            ]
    
    async def replay_scenarios(self, duration: int = 300) -> TestMetrics:
        """Replay production scenarios"""
        metrics = TestMetrics()
        scenario_runner = QuantLibTestScenario()
        
        start_time = time.time()
        scenario_index = 0
        
        logger.info(f"Replaying {len(self.scenarios)} production scenarios")
        
        while time.time() - start_time < duration and scenario_index < len(self.scenarios):
            scenario = self.scenarios[scenario_index]
            operation_start = time.time()
            
            try:
                if scenario['operation_type'] == 'option_pricing':
                    params = scenario['parameters']
                    option = scenario_runner.create_european_option(
                        params['strike'], 
                        params.get('maturity', 30),
                        params.get('option_type', 'Call')
                    )
                    result = option.NPV()
                    
                elif scenario['operation_type'] == 'portfolio_analysis':
                    # Simulate portfolio analysis
                    portfolio_size = scenario['parameters'].get('portfolio_size', 50)
                    for _ in range(min(portfolio_size, 20)):  # Limit for performance
                        option = scenario_runner.create_european_option(
                            random.uniform(80, 120), 30, 'Call'
                        )
                        result = option.NPV()
                
                metrics.successful_operations += 1
                
                response_time = time.time() - operation_start
                metrics.average_response_time = (
                    (metrics.average_response_time * (metrics.successful_operations - 1) + response_time) 
                    / metrics.successful_operations
                )
                
            except Exception as e:
                metrics.failed_operations += 1
                logger.error(f"Scenario replay failed: {e}")
            
            metrics.total_operations += 1
            scenario_index += 1
            
            # Small delay between scenarios
            await asyncio.sleep(0.01)
        
        metrics.error_rate = metrics.failed_operations / max(metrics.total_operations, 1)
        metrics.throughput = metrics.total_operations / duration
        
        return metrics

class QuantLibProductionTestHarness:
    """Main test harness for QuantLib production scenarios"""
    
    def __init__(self):
        self.patterns: Dict[str, QuantLibUsagePattern] = {}
        self.results: Dict[str, TestMetrics] = {}
        self.data_driven = DataDrivenScenarios()
        self.ramp_tester = None
    
    def register_pattern(self, name: str, pattern: QuantLibUsagePattern):
        """Register a usage pattern"""
        self.patterns[name] = pattern
    
    def setup_default_patterns(self):
        """Setup default QuantLib patterns"""
        self.register_pattern("hft_trading", HighFrequencyTradingPattern(requests_per_second=50))
        self.register_pattern("portfolio_risk", PortfolioRiskAnalysisPattern(portfolio_size=100))
        self.register_pattern("vol_calibration", VolatilityCalibrationPattern(calibration_points=30))
        self.register_pattern("market_data_feed", MarketDataFeedPattern(feed_rate=20))
    
    async def run_pattern(self, pattern_name: str, duration: int = 60) -> TestMetrics:
        """Run a specific usage pattern"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Pattern '{pattern_name}' not registered")
        
        logger.info(f"Starting QuantLib pattern: {pattern_name} for {duration} seconds")
        start_time = time.time()
        
        pattern = self.patterns[pattern_name]
        metrics = await pattern.execute(duration)
        
        actual_duration = time.time() - start_time
        logger.info(f"Pattern '{pattern_name}' completed in {actual_duration:.2f}s")
        
        self.results[pattern_name] = metrics
        return metrics
    
    async def run_all_patterns(self, duration: int = 60) -> Dict[str, TestMetrics]:
        """Run all registered patterns"""
        results = {}
        
        for pattern_name in self.patterns:
            try:
                metrics = await self.run_pattern(pattern_name, duration)
                results[pattern_name] = metrics
            except Exception as e:
                logger.error(f"Pattern '{pattern_name}' failed: {e}")
                results[pattern_name] = TestMetrics()
        
        return results
    
    async def run_endurance_test(self, pattern_name: str, duration: int = 3600) -> TestMetrics:
        """Run long-running endurance test"""
        logger.info(f"Starting endurance test for {pattern_name} - Duration: {duration}s ({duration//3600}h)")
        
        if pattern_name not in self.patterns:
            raise ValueError(f"Pattern '{pattern_name}' not registered")
        
        pattern = self.patterns[pattern_name]
        
        # Monitor for memory leaks and performance degradation
        checkpoint_interval = duration // 10  # 10 checkpoints
        checkpoints = []
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                checkpoint_start = time.time()
                
                # Run pattern for checkpoint interval
                metrics = await pattern.execute(min(checkpoint_interval, duration - (time.time() - start_time)))
                
                checkpoint_duration = time.time() - checkpoint_start
                checkpoints.append({
                    'elapsed_time': time.time() - start_time,
                    'metrics': asdict(metrics),
                    'checkpoint_duration': checkpoint_duration
                })
                
                logger.info(f"Endurance checkpoint - Elapsed: {time.time() - start_time:.0f}s, "
                           f"Memory: {metrics.peak_memory_usage:.1f}MB, "
                           f"Error Rate: {metrics.error_rate:.2%}")
                
                # Check for performance degradation
                if len(checkpoints) > 1:
                    current_throughput = metrics.throughput
                    initial_throughput = checkpoints[0]['metrics']['throughput']
                    
                    if current_throughput < initial_throughput * 0.7:  # 30% degradation
                        logger.warning("Performance degradation detected during endurance test")
                
                # Brief pause between checkpoints
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Endurance test failed: {e}")
            raise
        
        # Aggregate final metrics
        final_metrics = TestMetrics()
        for checkpoint in checkpoints:
            cp_metrics = checkpoint['metrics']
            final_metrics.total_operations += cp_metrics['total_operations']
            final_metrics.successful_operations += cp_metrics['successful_operations']
            final_metrics.failed_operations += cp_metrics['failed_operations']
            final_metrics.peak_memory_usage = max(final_metrics.peak_memory_usage, cp_metrics['peak_memory_usage'])
        
        final_metrics.error_rate = final_metrics.failed_operations / max(final_metrics.total_operations, 1)
        final_metrics.throughput = final_metrics.total_operations / duration
        
        logger.info(f"Endurance test completed - Total ops: {final_metrics.total_operations}, "
                   f"Error rate: {final_metrics.error_rate:.2%}")
        
        return final_metrics, checkpoints
    
    async def run_load_ramp_test(self, pattern_class, max_load: int = 100) -> Dict[str, Any]:
        """Run gradual load ramping test"""
        self.ramp_tester = GradualLoadRamping(pattern_class, max_load=max_load)
        return await self.ramp_tester.execute()
    
    async def run_data_driven_scenarios(self, duration: int = 300) -> TestMetrics:
        """Run data-driven scenarios based on production logs"""
        self.data_driven.load_production_logs()  # Load sample data
        return await self.data_driven.replay_scenarios(duration)
    
    async def run_comprehensive_test_suite(self, short_duration: int = 60, long_duration: int = 300) -> Dict[str, Any]:
        """Run comprehensive test suite with all advanced features"""
        logger.info("Starting comprehensive QuantLib test suite")
        
        suite_results = {
            'basic_patterns': {},
            'endurance_tests': {},
            'load_ramp_tests': {},
            'data_driven_results': None,
            'chaos_engineering': {},
            'resource_analysis': {},
            'summary': {}
        }
        
        # 1. Basic pattern tests
        logger.info("Phase 1: Basic pattern tests")
        suite_results['basic_patterns'] = await self.run_all_patterns(short_duration)
        
        # 2. Endurance tests (shorter for demo)
        logger.info("Phase 2: Endurance tests")
        for pattern_name in ['hft_trading', 'portfolio_risk']:
            try:
                endurance_metrics, checkpoints = await self.run_endurance_test(pattern_name, long_duration)
                suite_results['endurance_tests'][pattern_name] = {
                    'metrics': asdict(endurance_metrics),
                    'checkpoints': checkpoints
                }
            except Exception as e:
                logger.error(f"Endurance test failed for {pattern_name}: {e}")
        
        # 3. Load ramp tests
        logger.info("Phase 3: Load ramping tests")
        for pattern_class in [HighFrequencyTradingPattern, PortfolioRiskAnalysisPattern]:
            try:
                ramp_results = await self.run_load_ramp_test(pattern_class, max_load=50)
                suite_results['load_ramp_tests'][pattern_class.__name__] = ramp_results
            except Exception as e:
                logger.error(f"Load ramp test failed for {pattern_class.__name__}: {e}")
        
        # 4. Data-driven scenarios
        logger.info("Phase 4: Data-driven scenarios")
        try:
            suite_results['data_driven_results'] = asdict(await self.run_data_driven_scenarios(short_duration))
        except Exception as e:
            logger.error(f"Data-driven scenarios failed: {e}")
        
        # 5. Generate summary
        suite_results['summary'] = self._generate_test_summary(suite_results)
        
        logger.info("Comprehensive test suite completed")
        return suite_results
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            'total_tests_run': 0,
            'total_operations': 0,
            'overall_success_rate': 0.0,
            'performance_issues': [],
            'resource_issues': [],
            'recommendations': []
        }
        
        # Analyze basic patterns
        if results['basic_patterns']:
            for pattern_name, metrics in results['basic_patterns'].items():
                if isinstance(metrics, TestMetrics):
                    summary['total_tests_run'] += 1
                    summary['total_operations'] += metrics.total_operations
                    
                    # Check for issues
                    if metrics.error_rate > 0.05:
                        summary['performance_issues'].append(f"{pattern_name}: High error rate ({metrics.error_rate:.2%})")
                    
                    if metrics.average_response_time > 1.0:
                        summary['performance_issues'].append(f"{pattern_name}: Slow response time ({metrics.average_response_time:.3f}s)")
                    
                    if metrics.peak_memory_usage > 1000:  # 1GB
                        summary['resource_issues'].append(f"{pattern_name}: High memory usage ({metrics.peak_memory_usage:.1f}MB)")
        
        # Analyze load ramp tests
        if results['load_ramp_tests']:
            for pattern_name, ramp_data in results['load_ramp_tests'].items():
                if ramp_data.get('breaking_point'):
                    summary['recommendations'].append(
                        f"{pattern_name}: Breaking point at load level {ramp_data['breaking_point']}"
                    )
        
        # Calculate overall success rate
        total_successful = sum(
            metrics.successful_operations for metrics in results['basic_patterns'].values()
            if isinstance(metrics, TestMetrics)
        )
        
        if summary['total_operations'] > 0:
            summary['overall_success_rate'] = total_successful / summary['total_operations']
        
        return summary
    
    def export_results(self, filename: str = "quantlib_test_results.json"):
        """Export test results to JSON file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'results': {
                name: asdict(metrics) if isinstance(metrics, TestMetrics) else metrics
                for name, metrics in self.results.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Results exported to {filename}")
    
    def export_csv_report(self, filename: str = "quantlib_test_report.csv"):
        """Export test results to CSV file"""
        csv_data = []
        
        for pattern_name, metrics in self.results.items():
            if isinstance(metrics, TestMetrics):
                csv_data.append({
                    'Pattern': pattern_name,
                    'Total Operations': metrics.total_operations,
                    'Success Rate': f"{(1 - metrics.error_rate):.2%}",
                    'Throughput (ops/sec)': f"{metrics.throughput:.2f}",
                    'Avg Response Time (s)': f"{metrics.average_response_time:.3f}",
                    'Peak Memory (MB)': f"{metrics.peak_memory_usage:.1f}",
                    'Cache Hit Rate': f"{metrics.cache_hits / max(metrics.cache_hits + metrics.cache_misses, 1):.2%}",
                    'Pricing Errors': metrics.pricing_errors,
                    'Calculation Errors': metrics.calculation_errors
                })
        
        if csv_data:
            fieldnames = csv_data[0].keys()
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            logger.info(f"CSV report exported to {filename}")
    
    def print_detailed_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("QUANTLIB PRODUCTION TEST HARNESS - DETAILED RESULTS")
        print("="*80)
        
        for pattern_name, metrics in self.results.items():
            if isinstance(metrics, TestMetrics):
                print(f"\n📊 Pattern: {pattern_name.upper()}")
                print(f"   {'='*50}")
                print(f"   Operations: {metrics.total_operations:,} total | "
                      f"{metrics.successful_operations:,} successful | "
                      f"{metrics.failed_operations:,} failed")
                print(f"   Success Rate: {(1-metrics.error_rate):.2%}")
                print(f"   Performance: {metrics.throughput:.2f} ops/sec | "
                      f"{metrics.average_response_time:.3f}s avg response")
                print(f"   Response Times: {metrics.min_response_time:.3f}s min | "
                      f"{metrics.max_response_time:.3f}s max")
                print(f"   Memory Usage: {metrics.peak_memory_usage:.1f}MB peak | "
                      f"{metrics.avg_memory_usage:.1f}MB average")
                print(f"   CPU Usage: {metrics.cpu_usage_peak:.1f}% peak")
                print(f"   Cache Performance: {metrics.cache_hits:,} hits | "
                      f"{metrics.cache_misses:,} misses | "
                      f"{metrics.cache_hits/(max(metrics.cache_hits + metrics.cache_misses, 1)):.2%} hit rate")
                print(f"   Errors: {metrics.pricing_errors} pricing | "
                      f"{metrics.calculation_errors} calculation | "
                      f"{metrics.convergence_failures} convergence")
                print(f"   GC Collections: {metrics.gc_collections}")
        
        print(f"\n{'='*80}")
        print("Test completed successfully! 🎉")

# Example usage and main execution
async def main():
    """Main execution function demonstrating QuantLib test harness"""
    
    if not QUANTLIB_AVAILABLE:
        print("⚠️  QuantLib not available - using mock implementation for demonstration")
        print("   Install QuantLib-Python for full functionality")
    
    print("🚀 Starting QuantLib Production Test Harness")
    print("   Testing financial computation patterns with advanced features")
    
    # Create test harness
    harness = QuantLibProductionTestHarness()
    
    # Setup default patterns
    harness.setup_default_patterns()
    
    # Run comprehensive test suite
    try:
        # Quick test run (shorter durations for demo)
        suite_results = await harness.run_comprehensive_test_suite(
            short_duration=30,  # 30 seconds for basic tests
            long_duration=60    # 1 minute for endurance tests
        )
        
        # Print detailed results
        harness.print_detailed_results()
        
        # Print summary
        summary = suite_results['summary']
        print(f"\n📈 EXECUTIVE SUMMARY")
        print(f"   {'='*40}")
        print(f"   Tests Run: {summary['total_tests_run']}")
        print(f"   Total Operations: {summary['total_operations']:,}")
        print(f"   Overall Success Rate: {summary['overall_success_rate']:.2%}")
        
        if summary['performance_issues']:
            print(f"   ⚠️  Performance Issues:")
            for issue in summary['performance_issues']:
                print(f"      - {issue}")
        
        if summary['resource_issues']:
            print(f"   💾 Resource Issues:")
            for issue in summary['resource_issues']:
                print(f"      - {issue}")
        
        if summary['recommendations']:
            print(f"   💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"      - {rec}")
        
        # Export results
        harness.export_results("quantlib_test_results.json")
        harness.export_csv_report("quantlib_test_report.csv")
        
    except Exception as e:
        logger.error(f"Test harness execution failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())