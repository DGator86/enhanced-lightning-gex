"""
Lightning GEX - Multi-Agent Coordination System
Specialized agents with consensus-building mechanism

Agent Specialization:
1. Trend Agent: Long-term directional bias
2. Reversal Agent: Mean reversion / oversold/overbought
3. Breakout Agent: Support/resistance breaks
4. Volatility Agent: Regime changes and volatility expansion
5. Options Flow Agent: GEX/Charm/Vanna (existing Agent 4)

Consensus Mechanism:
- Weighted voting based on agent confidence + historical performance
- Conflict resolution when agents disagree
- Dynamic weight adjustment based on recent accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class AgentSignal:
    """Signal from a single specialized agent"""
    
    agent_name: str
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0 to 1
    reasoning: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: int = 7  # Days
    supporting_indicators: List[str] = None
    
    def __post_init__(self):
        if self.supporting_indicators is None:
            self.supporting_indicators = []


class TrendAgent:
    """
    Specialized in identifying and following long-term trends
    Uses: Moving averages, trend strength, momentum
    """
    
    def __init__(self):
        self.name = "Trend Agent"
        self.performance_history = []
        
    def analyze(self, data: Dict) -> AgentSignal:
        """Analyze trend direction and strength"""
        
        # Extract relevant data
        price = data.get('price', 0)
        sma_20 = data.get('sma_20', price)
        sma_50 = data.get('sma_50', price)
        sma_200 = data.get('sma_200', price)
        adx = data.get('adx', 25)  # Trend strength indicator
        
        # Determine trend direction
        bullish_signals = 0
        bearish_signals = 0
        
        if price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if price > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if sma_20 > sma_50 > sma_200:
            bullish_signals += 2  # Strong bullish alignment
        elif sma_20 < sma_50 < sma_200:
            bearish_signals += 2  # Strong bearish alignment
        
        # Determine direction
        if bullish_signals > bearish_signals:
            direction = 'bullish'
            signal_strength = bullish_signals / (bullish_signals + bearish_signals)
        elif bearish_signals > bullish_signals:
            direction = 'bearish'
            signal_strength = bearish_signals / (bullish_signals + bearish_signals)
        else:
            direction = 'neutral'
            signal_strength = 0.5
        
        # Adjust confidence based on trend strength (ADX)
        if adx > 40:
            trend_multiplier = 1.2  # Strong trend
        elif adx > 25:
            trend_multiplier = 1.0  # Moderate trend
        else:
            trend_multiplier = 0.7  # Weak trend
        
        confidence = min(signal_strength * trend_multiplier, 1.0)
        
        # Generate reasoning
        reasoning = f"Price vs SMAs: {bullish_signals}/{bearish_signals}, ADX: {adx:.1f}"
        
        # Calculate targets
        if direction == 'bullish':
            target_price = price * 1.05  # 5% upside
            stop_loss = sma_20 * 0.98  # Below 20-day SMA
        elif direction == 'bearish':
            target_price = price * 0.95  # 5% downside
            stop_loss = sma_20 * 1.02  # Above 20-day SMA
        else:
            target_price = price
            stop_loss = price * 0.98
        
        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon=14,  # Trend plays need time
            supporting_indicators=['SMA', 'ADX']
        )


class ReversalAgent:
    """
    Specialized in mean reversion and identifying oversold/overbought conditions
    Uses: RSI, Bollinger Bands, Stochastics
    """
    
    def __init__(self):
        self.name = "Reversal Agent"
        self.performance_history = []
        
    def analyze(self, data: Dict) -> AgentSignal:
        """Analyze reversal potential"""
        
        # Extract relevant data
        price = data.get('price', 0)
        rsi = data.get('rsi', 50)
        bb_upper = data.get('bb_upper', price * 1.02)
        bb_lower = data.get('bb_lower', price * 0.98)
        stoch = data.get('stochastic', 50)
        
        # Determine reversal signals
        oversold_signals = 0
        overbought_signals = 0
        
        # RSI analysis
        if rsi < 30:
            oversold_signals += 2  # Strong oversold
        elif rsi < 40:
            oversold_signals += 1  # Mild oversold
        elif rsi > 70:
            overbought_signals += 2  # Strong overbought
        elif rsi > 60:
            overbought_signals += 1  # Mild overbought
        
        # Bollinger Bands
        if price < bb_lower:
            oversold_signals += 1
        elif price > bb_upper:
            overbought_signals += 1
        
        # Stochastics
        if stoch < 20:
            oversold_signals += 1
        elif stoch > 80:
            overbought_signals += 1
        
        # Determine direction (reversal means opposite of current extreme)
        if oversold_signals > overbought_signals:
            direction = 'bullish'  # Expecting bounce from oversold
            signal_strength = oversold_signals / 5  # Max 5 signals
        elif overbought_signals > oversold_signals:
            direction = 'bearish'  # Expecting pullback from overbought
            signal_strength = overbought_signals / 5
        else:
            direction = 'neutral'
            signal_strength = 0.3
        
        confidence = min(signal_strength, 0.9)
        
        # Generate reasoning
        reasoning = f"RSI: {rsi:.1f}, Stoch: {stoch:.1f}, BB Position: {((price - bb_lower) / (bb_upper - bb_lower)):.1%}"
        
        # Calculate targets (smaller moves for reversions)
        if direction == 'bullish':
            target_price = price * 1.03  # 3% bounce
            stop_loss = bb_lower * 0.98
        elif direction == 'bearish':
            target_price = price * 0.97  # 3% pullback
            stop_loss = bb_upper * 1.02
        else:
            target_price = price
            stop_loss = price * 0.98
        
        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon=5,  # Reversals happen quickly
            supporting_indicators=['RSI', 'Bollinger Bands', 'Stochastic']
        )


class BreakoutAgent:
    """
    Specialized in detecting support/resistance breaks
    Uses: Volume, price action, volatility
    """
    
    def __init__(self):
        self.name = "Breakout Agent"
        self.performance_history = []
        
    def analyze(self, data: Dict) -> AgentSignal:
        """Analyze breakout potential"""
        
        # Extract relevant data
        price = data.get('price', 0)
        resistance = data.get('resistance', price * 1.02)
        support = data.get('support', price * 0.98)
        volume = data.get('volume', 1000000)
        avg_volume = data.get('avg_volume', volume)
        atr = data.get('atr', price * 0.02)
        
        # Calculate distance to levels
        dist_to_resistance = (resistance - price) / price
        dist_to_support = (price - support) / price
        
        # Volume confirmation
        volume_multiplier = volume / avg_volume
        
        # Determine breakout potential
        if dist_to_resistance < 0.005 and volume_multiplier > 1.5:
            # Near resistance with high volume - bullish breakout
            direction = 'bullish'
            confidence = min(0.70 + (volume_multiplier - 1.5) * 0.1, 0.95)
            reasoning = f"Near resistance (${resistance:.2f}), volume {volume_multiplier:.1f}x avg"
            target_price = resistance + (atr * 2)  # 2 ATRs above resistance
            stop_loss = resistance * 0.995  # Tight stop below resistance
            
        elif dist_to_support < 0.005 and volume_multiplier > 1.5:
            # Near support with high volume - bearish breakdown
            direction = 'bearish'
            confidence = min(0.70 + (volume_multiplier - 1.5) * 0.1, 0.95)
            reasoning = f"Near support (${support:.2f}), volume {volume_multiplier:.1f}x avg"
            target_price = support - (atr * 2)  # 2 ATRs below support
            stop_loss = support * 1.005  # Tight stop above support
            
        elif abs(dist_to_resistance) < 0.02 or abs(dist_to_support) < 0.02:
            # Near levels but no volume confirmation
            direction = 'neutral'
            confidence = 0.3
            reasoning = "Near key levels but no volume confirmation"
            target_price = price
            stop_loss = price * 0.98
        else:
            # No breakout setup
            direction = 'neutral'
            confidence = 0.2
            reasoning = "No breakout setup - price in middle of range"
            target_price = price
            stop_loss = price * 0.98
        
        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon=3,  # Breakouts happen fast
            supporting_indicators=['Volume', 'Price Action', 'ATR']
        )


class VolatilityAgent:
    """
    Specialized in volatility regime changes
    Uses: ATR, Bollinger Band width, historical volatility
    """
    
    def __init__(self):
        self.name = "Volatility Agent"
        self.performance_history = []
        
    def analyze(self, data: Dict) -> AgentSignal:
        """Analyze volatility conditions"""
        
        # Extract relevant data
        price = data.get('price', 0)
        atr = data.get('atr', price * 0.02)
        atr_20_avg = data.get('atr_20_avg', atr)
        bb_width = data.get('bb_width', 0.04)
        bb_width_avg = data.get('bb_width_avg', bb_width)
        
        # Calculate volatility expansion/contraction
        atr_ratio = atr / atr_20_avg if atr_20_avg > 0 else 1.0
        bb_ratio = bb_width / bb_width_avg if bb_width_avg > 0 else 1.0
        
        # Determine volatility regime
        if atr_ratio < 0.7 and bb_ratio < 0.7:
            # Low volatility - expect expansion (typically bullish for markets)
            direction = 'bullish'
            confidence = 0.65
            reasoning = f"Volatility compression (ATR: {atr_ratio:.2f}x, BB: {bb_ratio:.2f}x) - expect expansion"
            time_horizon = 7
            
        elif atr_ratio > 1.5 and bb_ratio > 1.5:
            # High volatility - expect mean reversion (bearish)
            direction = 'bearish'
            confidence = 0.60
            reasoning = f"Volatility spike (ATR: {atr_ratio:.2f}x, BB: {bb_ratio:.2f}x) - expect reversion"
            time_horizon = 3
            
        else:
            # Normal volatility
            direction = 'neutral'
            confidence = 0.3
            reasoning = f"Normal volatility regime (ATR: {atr_ratio:.2f}x, BB: {bb_ratio:.2f}x)"
            time_horizon = 5
        
        # Calculate targets
        if direction == 'bullish':
            target_price = price * 1.04
            stop_loss = price * 0.97
        elif direction == 'bearish':
            target_price = price * 0.96
            stop_loss = price * 1.03
        else:
            target_price = price
            stop_loss = price * 0.98
        
        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon=time_horizon,
            supporting_indicators=['ATR', 'Bollinger Width', 'Volatility Regime']
        )


class OptionsFlowAgent:
    """
    Specialized in options flow and Greek positioning
    This is your existing Enhanced Agent 4 with GEX/Charm/Vanna
    """
    
    def __init__(self):
        self.name = "Options Flow Agent"
        self.performance_history = []
        
    def analyze(self, data: Dict) -> AgentSignal:
        """Analyze options positioning"""
        
        # Extract GEX/Charm/Vanna data
        gex_signal = data.get('gex_signal', 0)
        charm_pressure = data.get('charm_pressure', 0)
        vanna_sensitivity = data.get('vanna_sensitivity', 0)
        dark_pool_flow = data.get('dark_pool_flow', 0)
        
        price = data.get('price', 0)
        
        # Combine signals
        signal_strength = (gex_signal + charm_pressure + vanna_sensitivity + dark_pool_flow) / 4
        
        # Determine direction
        if signal_strength > 0.3:
            direction = 'bullish'
            confidence = min(abs(signal_strength) * 1.2, 0.95)
        elif signal_strength < -0.3:
            direction = 'bearish'
            confidence = min(abs(signal_strength) * 1.2, 0.95)
        else:
            direction = 'neutral'
            confidence = 0.4
        
        # Generate reasoning
        reasoning = f"GEX: {gex_signal:.2f}, Charm: {charm_pressure:.2f}, Vanna: {vanna_sensitivity:.2f}, DP: {dark_pool_flow:.2f}"
        
        # Calculate targets
        if direction == 'bullish':
            target_price = price * 1.05
            stop_loss = price * 0.97
        elif direction == 'bearish':
            target_price = price * 0.95
            stop_loss = price * 1.03
        else:
            target_price = price
            stop_loss = price * 0.98
        
        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon=7,
            supporting_indicators=['GEX', 'Charm', 'Vanna', 'Dark Pool']
        )


class ConsensusBuilder:
    """
    Builds consensus from multiple agent signals
    Uses weighted voting based on confidence + historical performance
    """
    
    def __init__(self):
        self.agent_weights = {
            'Trend Agent': 1.0,
            'Reversal Agent': 1.0,
            'Breakout Agent': 1.0,
            'Volatility Agent': 1.0,
            'Options Flow Agent': 1.5  # Higher weight for GEX/Charm/Vanna
        }
    
    def build_consensus(self, signals: List[AgentSignal]) -> Dict:
        """
        Build consensus from agent signals
        
        Returns final decision with reasoning
        """
        
        if not signals:
            return self._neutral_consensus()
        
        # Calculate weighted votes
        bullish_weight = 0
        bearish_weight = 0
        neutral_weight = 0
        
        total_weight = 0
        
        for signal in signals:
            agent_weight = self.agent_weights.get(signal.agent_name, 1.0)
            vote_weight = signal.confidence * agent_weight
            
            if signal.direction == 'bullish':
                bullish_weight += vote_weight
            elif signal.direction == 'bearish':
                bearish_weight += vote_weight
            else:
                neutral_weight += vote_weight
            
            total_weight += agent_weight
        
        # Determine consensus
        if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
            direction = 'bullish'
            confidence = bullish_weight / total_weight
            agreement = bullish_weight / (bullish_weight + bearish_weight + neutral_weight)
        elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
            direction = 'bearish'
            confidence = bearish_weight / total_weight
            agreement = bearish_weight / (bullish_weight + bearish_weight + neutral_weight)
        else:
            direction = 'neutral'
            confidence = 0.4
            agreement = neutral_weight / (bullish_weight + bearish_weight + neutral_weight) if (bullish_weight + bearish_weight + neutral_weight) > 0 else 0
        
        # Calculate consensus metrics
        supporting_agents = [s for s in signals if s.direction == direction]
        conflicting_agents = [s for s in signals if s.direction != direction and s.direction != 'neutral']
        
        # Aggregate targets (weighted average)
        if supporting_agents:
            total_conf = sum(s.confidence for s in supporting_agents)
            target_price = sum(s.target_price * s.confidence for s in supporting_agents if s.target_price) / total_conf if total_conf > 0 else None
            stop_loss = sum(s.stop_loss * s.confidence for s in supporting_agents if s.stop_loss) / total_conf if total_conf > 0 else None
            avg_horizon = int(np.mean([s.time_horizon for s in supporting_agents]))
        else:
            target_price = None
            stop_loss = None
            avg_horizon = 7
        
        # Build consensus result
        consensus = {
            'direction': direction,
            'confidence': confidence,
            'agreement': agreement,  # % of votes agreeing
            'target_price': target_price,
            'stop_loss': stop_loss,
            'time_horizon': avg_horizon,
            'supporting_agents': [s.agent_name for s in supporting_agents],
            'conflicting_agents': [s.agent_name for s in conflicting_agents],
            'agent_signals': [
                {
                    'agent': s.agent_name,
                    'direction': s.direction,
                    'confidence': s.confidence,
                    'reasoning': s.reasoning
                } for s in signals
            ],
            'consensus_strength': self._assess_strength(agreement, len(supporting_agents), len(signals)),
            'timestamp': datetime.now().isoformat()
        }
        
        return consensus
    
    def _neutral_consensus(self) -> Dict:
        """Return neutral consensus when no signals"""
        return {
            'direction': 'neutral',
            'confidence': 0.3,
            'agreement': 0,
            'target_price': None,
            'stop_loss': None,
            'time_horizon': 7,
            'supporting_agents': [],
            'conflicting_agents': [],
            'agent_signals': [],
            'consensus_strength': 'WEAK',
            'timestamp': datetime.now().isoformat()
        }
    
    def _assess_strength(self, agreement: float, supporting: int, total: int) -> str:
        """Assess consensus strength"""
        
        if agreement >= 0.80 and supporting >= 4:
            return 'VERY STRONG'
        elif agreement >= 0.70 and supporting >= 3:
            return 'STRONG'
        elif agreement >= 0.60:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def update_agent_weight(self, agent_name: str, performance: float):
        """
        Update agent weight based on recent performance
        
        performance: Win rate (0 to 1)
        """
        
        if agent_name in self.agent_weights:
            # Adjust weight: 0.5x to 1.5x based on performance
            adjustment = 0.5 + (performance * 1.0)
            self.agent_weights[agent_name] *= adjustment
            
            # Keep weights in reasonable range
            self.agent_weights[agent_name] = np.clip(self.agent_weights[agent_name], 0.5, 2.0)


class LightningMultiAgentCoordinator:
    """
    Main coordinator for multi-agent system
    """
    
    def __init__(self):
        # Initialize agents
        self.trend_agent = TrendAgent()
        self.reversal_agent = ReversalAgent()
        self.breakout_agent = BreakoutAgent()
        self.volatility_agent = VolatilityAgent()
        self.options_agent = OptionsFlowAgent()
        
        # Initialize consensus builder
        self.consensus_builder = ConsensusBuilder()
        
        # Track decisions
        self.decision_history = []
    
    def analyze(self, data: Dict) -> Dict:
        """
        Run full multi-agent analysis
        
        Returns consensus decision
        """
        
        print(f"\n{'='*80}")
        print("MULTI-AGENT ANALYSIS")
        print(f"{'='*80}\n")
        
        # Collect signals from all agents
        signals = []
        
        print("Collecting Agent Signals...")
        
        # Trend Agent
        trend_signal = self.trend_agent.analyze(data)
        signals.append(trend_signal)
        print(f"  {trend_signal.agent_name}: {trend_signal.direction.upper()} ({trend_signal.confidence:.1%})")
        
        # Reversal Agent
        reversal_signal = self.reversal_agent.analyze(data)
        signals.append(reversal_signal)
        print(f"  {reversal_signal.agent_name}: {reversal_signal.direction.upper()} ({reversal_signal.confidence:.1%})")
        
        # Breakout Agent
        breakout_signal = self.breakout_agent.analyze(data)
        signals.append(breakout_signal)
        print(f"  {breakout_signal.agent_name}: {breakout_signal.direction.upper()} ({breakout_signal.confidence:.1%})")
        
        # Volatility Agent
        vol_signal = self.volatility_agent.analyze(data)
        signals.append(vol_signal)
        print(f"  {vol_signal.agent_name}: {vol_signal.direction.upper()} ({vol_signal.confidence:.1%})")
        
        # Options Flow Agent
        options_signal = self.options_agent.analyze(data)
        signals.append(options_signal)
        print(f"  {options_signal.agent_name}: {options_signal.direction.upper()} ({options_signal.confidence:.1%})")
        
        # Build consensus
        print(f"\nBuilding Consensus...")
        consensus = self.consensus_builder.build_consensus(signals)
        
        print(f"\n{'='*80}")
        print(f"CONSENSUS: {consensus['direction'].upper()}")
        print(f"{'='*80}")
        print(f"Confidence: {consensus['confidence']:.1%}")
        print(f"Agreement: {consensus['agreement']:.1%}")
        print(f"Strength: {consensus['consensus_strength']}")
        print(f"Supporting: {', '.join(consensus['supporting_agents'])}")
        if consensus['conflicting_agents']:
            print(f"Conflicting: {', '.join(consensus['conflicting_agents'])}")
        print(f"{'='*80}\n")
        
        # Record decision
        self.decision_history.append(consensus)
        
        return consensus
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of multi-agent system"""
        
        if not self.decision_history:
            return {'status': 'no_history'}
        
        # Calculate metrics
        total_decisions = len(self.decision_history)
        
        # Count by direction
        bullish = sum(1 for d in self.decision_history if d['direction'] == 'bullish')
        bearish = sum(1 for d in self.decision_history if d['direction'] == 'bearish')
        neutral = sum(1 for d in self.decision_history if d['direction'] == 'neutral')
        
        # Average confidence and agreement
        avg_confidence = np.mean([d['confidence'] for d in self.decision_history])
        avg_agreement = np.mean([d['agreement'] for d in self.decision_history])
        
        # Strength distribution
        strengths = [d['consensus_strength'] for d in self.decision_history]
        strength_counts = {s: strengths.count(s) for s in set(strengths)}
        
        return {
            'total_decisions': total_decisions,
            'by_direction': {
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral
            },
            'avg_confidence': avg_confidence,
            'avg_agreement': avg_agreement,
            'by_strength': strength_counts,
            'recent_decisions': self.decision_history[-5:]
        }


def demo_multi_agent_coordinator():
    """Demo showing multi-agent coordination"""
    
    print("\n" + "="*80)
    print("LIGHTNING GEX - MULTI-AGENT COORDINATOR DEMO")
    print("="*80)
    
    # Create coordinator
    coordinator = LightningMultiAgentCoordinator()
    
    # Test scenario 1: Strong bullish consensus
    print("\nScenario 1: Strong Bullish Setup")
    data1 = {
        'price': 450.0,
        'sma_20': 445.0,
        'sma_50': 440.0,
        'sma_200': 430.0,
        'adx': 45.0,
        'rsi': 55.0,
        'bb_upper': 455.0,
        'bb_lower': 445.0,
        'stochastic': 60.0,
        'resistance': 452.0,
        'support': 448.0,
        'volume': 1500000,
        'avg_volume': 1000000,
        'atr': 4.5,
        'atr_20_avg': 5.0,
        'bb_width': 0.022,
        'bb_width_avg': 0.025,
        'gex_signal': 0.8,
        'charm_pressure': 0.7,
        'vanna_sensitivity': 0.6,
        'dark_pool_flow': 0.7
    }
    
    consensus1 = coordinator.analyze(data1)
    
    # Test scenario 2: Conflicting signals
    print("\n" + "="*80)
    print("\nScenario 2: Conflicting Signals")
    data2 = {
        'price': 450.0,
        'sma_20': 452.0,  # Price below SMA (bearish)
        'sma_50': 445.0,
        'sma_200': 440.0,
        'adx': 25.0,
        'rsi': 25.0,  # Oversold (bullish reversal)
        'bb_upper': 455.0,
        'bb_lower': 445.0,
        'stochastic': 15.0,  # Oversold
        'resistance': 455.0,
        'support': 448.0,
        'volume': 900000,
        'avg_volume': 1000000,
        'atr': 3.0,  # Low volatility
        'atr_20_avg': 5.0,
        'bb_width': 0.015,
        'bb_width_avg': 0.025,
        'gex_signal': -0.3,  # Bearish
        'charm_pressure': 0.6,  # Bullish
        'vanna_sensitivity': -0.2,  # Bearish
        'dark_pool_flow': 0.4  # Slightly bullish
    }
    
    consensus2 = coordinator.analyze(data2)
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    summary = coordinator.get_performance_summary()
    print(f"\nTotal Decisions: {summary['total_decisions']}")
    print(f"Average Confidence: {summary['avg_confidence']:.1%}")
    print(f"Average Agreement: {summary['avg_agreement']:.1%}")
    print(f"\nBy Direction:")
    for direction, count in summary['by_direction'].items():
        print(f"  {direction.title()}: {count}")
    print(f"\nBy Strength:")
    for strength, count in summary['by_strength'].items():
        print(f"  {strength}: {count}")
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo_multi_agent_coordinator()
    
    print("="*80)
    print("INTEGRATION WITH LIGHTNING GEX")
    print("="*80)
    print("""
To integrate with your Lightning GEX system:

1. Initialize coordinator:
   
   coordinator = LightningMultiAgentCoordinator()

2. Prepare data from your system:
   
   data = {
       'price': current_price,
       'sma_20': sma_20,
       'sma_50': sma_50,
       'sma_200': sma_200,
       'adx': adx,
       'rsi': rsi,
       'bb_upper': bb_upper,
       'bb_lower': bb_lower,
       'stochastic': stochastic,
       'resistance': resistance_level,
       'support': support_level,
       'volume': current_volume,
       'avg_volume': avg_volume_20,
       'atr': atr,
       'atr_20_avg': atr_20_avg,
       'bb_width': bb_width,
       'bb_width_avg': bb_width_avg,
       'gex_signal': gex_signal,
       'charm_pressure': charm_pressure,
       'vanna_sensitivity': vanna_sensitivity,
       'dark_pool_flow': dark_pool_flow
   }

3. Get consensus decision:
   
   consensus = coordinator.analyze(data)
   
   if consensus['confidence'] >= 0.75 and consensus['consensus_strength'] in ['STRONG', 'VERY STRONG']:
       # High confidence trade
       execute_trade(
           direction=consensus['direction'],
           target=consensus['target_price'],
           stop=consensus['stop_loss']
       )

4. Update agent weights based on performance:
   
   coordinator.consensus_builder.update_agent_weight('Trend Agent', win_rate)

Expected Benefits:
- +2-5% accuracy improvement from multi-perspective analysis
- Better conflict resolution when signals disagree
- More robust decisions (not reliant on single methodology)
- Adaptive learning through agent weight adjustment
    """)
    print("="*80 + "\n")
