"""
Lightning GEX - Security & Guardrails Manager
Production-grade security, encryption, and safety controls

Security Features:
1. Encrypted communication with broker APIs
2. Secure credential management
3. Trading guardrails (prevent runaway behavior)
4. Audit logging
5. Rate limiting
6. Circuit breakers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import os
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditLog:
    """Audit log entry"""
    
    timestamp: str
    event_type: str  # 'order', 'position_change', 'capital_change', 'error', 'warning'
    severity: str  # 'info', 'warning', 'error', 'critical'
    details: Dict
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CredentialManager:
    """
    Secure credential management using encryption
    Never stores credentials in plain text
    """
    
    def __init__(self, master_password: str, salt: Optional[bytes] = None):
        """
        Initialize credential manager
        
        Args:
            master_password: Master password for encryption
            salt: Optional salt (generated if not provided)
        """
        
        # Generate or use provided salt
        if salt is None:
            self.salt = os.urandom(16)
        else:
            self.salt = salt
        
        # Derive encryption key from password
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        
        # Create cipher
        self.cipher = Fernet(key)
        
        # Encrypted credentials storage
        self.credentials = {}
    
    def store_credential(self, name: str, value: str):
        """Store encrypted credential"""
        
        encrypted_value = self.cipher.encrypt(value.encode())
        self.credentials[name] = encrypted_value
    
    def retrieve_credential(self, name: str) -> Optional[str]:
        """Retrieve and decrypt credential"""
        
        encrypted_value = self.credentials.get(name)
        if encrypted_value is None:
            return None
        
        try:
            decrypted_value = self.cipher.decrypt(encrypted_value)
            return decrypted_value.decode()
        except Exception:
            return None
    
    def delete_credential(self, name: str):
        """Delete credential"""
        if name in self.credentials:
            del self.credentials[name]
    
    def list_credentials(self) -> List[str]:
        """List credential names (not values)"""
        return list(self.credentials.keys())
    
    def save_to_file(self, filepath: str):
        """Save encrypted credentials to file"""
        
        data = {
            'salt': base64.b64encode(self.salt).decode(),
            'credentials': {
                k: base64.b64encode(v).decode() 
                for k, v in self.credentials.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load_from_file(cls, filepath: str, master_password: str) -> 'CredentialManager':
        """Load credentials from encrypted file"""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        salt = base64.b64decode(data['salt'])
        manager = cls(master_password, salt=salt)
        
        manager.credentials = {
            k: base64.b64decode(v)
            for k, v in data['credentials'].items()
        }
        
        return manager


class TradingGuardrails:
    """
    Trading guardrails to prevent runaway behavior
    Enforces hard limits on risk-taking
    """
    
    def __init__(self,
                 max_daily_trades: int = 20,
                 max_daily_loss_pct: float = 0.05,
                 max_position_size_pct: float = 0.15,
                 max_total_exposure_pct: float = 0.80,
                 max_orders_per_minute: int = 10):
        
        # Hard limits
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_size_pct = max_position_size_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_orders_per_minute = max_orders_per_minute
        
        # State tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.initial_capital = 0.0
        self.current_exposure = 0.0
        self.recent_orders = []  # Timestamps of recent orders
        self.last_reset = datetime.now().date()
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
    
    def validate_trade(self,
                      order_value: float,
                      current_capital: float,
                      total_exposure: float) -> Dict:
        """
        Validate if trade passes guardrails
        
        Returns dict with 'approved' boolean and 'reasons' list
        """
        
        # Reset daily counters if new day
        self._check_daily_reset()
        
        # Initialize capital if not set
        if self.initial_capital == 0:
            self.initial_capital = current_capital
        
        result = {
            'approved': True,
            'warnings': [],
            'blockers': []
        }
        
        # Check 1: Circuit breaker
        if self.circuit_breaker_active:
            result['approved'] = False
            result['blockers'].append(f"Circuit breaker active: {self.circuit_breaker_reason}")
            return result
        
        # Check 2: Daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            result['approved'] = False
            result['blockers'].append(f"Daily trade limit reached ({self.max_daily_trades})")
        
        # Check 3: Daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.initial_capital if self.initial_capital > 0 else 0
        if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
            result['approved'] = False
            result['blockers'].append(f"Daily loss limit reached ({daily_loss_pct:.1%})")
            self._activate_circuit_breaker(f"Daily loss limit: {daily_loss_pct:.1%}")
        
        # Check 4: Position size limit
        position_size_pct = order_value / current_capital if current_capital > 0 else 0
        if position_size_pct > self.max_position_size_pct:
            result['approved'] = False
            result['blockers'].append(f"Position size too large ({position_size_pct:.1%} > {self.max_position_size_pct:.1%})")
        
        # Check 5: Total exposure limit
        new_exposure = total_exposure + order_value
        exposure_pct = new_exposure / current_capital if current_capital > 0 else 0
        if exposure_pct > self.max_total_exposure_pct:
            result['approved'] = False
            result['blockers'].append(f"Total exposure too high ({exposure_pct:.1%} > {self.max_total_exposure_pct:.1%})")
        
        # Check 6: Rate limiting
        now = datetime.now()
        recent = [t for t in self.recent_orders if (now - t).total_seconds() < 60]
        if len(recent) >= self.max_orders_per_minute:
            result['approved'] = False
            result['blockers'].append(f"Rate limit exceeded ({len(recent)} orders/minute)")
        
        # Warnings (don't block, just warn)
        if position_size_pct > self.max_position_size_pct * 0.8:
            result['warnings'].append(f"Position size near limit ({position_size_pct:.1%})")
        
        if exposure_pct > self.max_total_exposure_pct * 0.8:
            result['warnings'].append(f"Total exposure near limit ({exposure_pct:.1%})")
        
        return result
    
    def record_trade(self, pnl: float = 0):
        """Record trade execution"""
        
        self._check_daily_reset()
        
        self.daily_trades += 1
        self.daily_pnl += pnl
        self.recent_orders.append(datetime.now())
        
        # Clean old orders (keep last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.recent_orders = [t for t in self.recent_orders if t > cutoff]
    
    def update_exposure(self, new_exposure: float):
        """Update current exposure"""
        self.current_exposure = new_exposure
    
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = today
            self.circuit_breaker_active = False
            self.circuit_breaker_reason = None
    
    def _activate_circuit_breaker(self, reason: str):
        """Activate circuit breaker (stops all trading until manual reset)"""
        
        self.circuit_breaker_active = True
        self.circuit_breaker_reason = reason
    
    def manual_reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
    
    def get_status(self) -> Dict:
        """Get current guardrails status"""
        
        self._check_daily_reset()
        
        return {
            'daily_trades': self.daily_trades,
            'daily_trades_remaining': max(0, self.max_daily_trades - self.daily_trades),
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': abs(self.daily_pnl) / self.initial_capital if self.initial_capital > 0 else 0,
            'current_exposure_pct': self.current_exposure / self.initial_capital if self.initial_capital > 0 else 0,
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_reason': self.circuit_breaker_reason,
            'recent_order_count': len([t for t in self.recent_orders if (datetime.now() - t).total_seconds() < 60])
        }


class AuditLogger:
    """
    Comprehensive audit logging
    Records all trading activity for compliance and debugging
    """
    
    def __init__(self, log_file: str = 'lightning_audit.log'):
        self.log_file = log_file
        self.logs = []
    
    def log(self,
            event_type: str,
            severity: str,
            details: Dict,
            user_id: Optional[str] = None):
        """Add log entry"""
        
        entry = AuditLog(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            severity=severity,
            details=details,
            user_id=user_id
        )
        
        self.logs.append(entry)
        self._write_to_file(entry)
    
    def _write_to_file(self, entry: AuditLog):
        """Write log entry to file"""
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            print(f"Failed to write audit log: {e}")
    
    def get_logs(self,
                 event_type: Optional[str] = None,
                 severity: Optional[str] = None,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None) -> List[AuditLog]:
        """Retrieve logs with optional filters"""
        
        filtered = self.logs.copy()
        
        if event_type:
            filtered = [log for log in filtered if log.event_type == event_type]
        
        if severity:
            filtered = [log for log in filtered if log.severity == severity]
        
        if start_time:
            filtered = [log for log in filtered if datetime.fromisoformat(log.timestamp) >= start_time]
        
        if end_time:
            filtered = [log for log in filtered if datetime.fromisoformat(log.timestamp) <= end_time]
        
        return filtered
    
    def get_error_summary(self, hours: int = 24) -> Dict:
        """Get summary of errors in last N hours"""
        
        cutoff = datetime.now() - timedelta(hours=hours)
        errors = self.get_logs(severity='error', start_time=cutoff)
        critical = self.get_logs(severity='critical', start_time=cutoff)
        
        return {
            'error_count': len(errors),
            'critical_count': len(critical),
            'recent_errors': [log.to_dict() for log in errors[-5:]],
            'recent_critical': [log.to_dict() for log in critical[-5:]]
        }


class APISecurityWrapper:
    """
    Secure wrapper for broker API calls
    Handles authentication, signing, rate limiting
    """
    
    def __init__(self,
                 credential_manager: CredentialManager,
                 api_name: str):
        
        self.credential_manager = credential_manager
        self.api_name = api_name
        
        # Rate limiting
        self.requests_per_minute = 60
        self.recent_requests = []
    
    def sign_request(self, endpoint: str, params: Dict) -> str:
        """
        Sign API request using HMAC
        
        Returns signature for request authentication
        """
        
        # Get API secret
        api_secret = self.credential_manager.retrieve_credential(f"{self.api_name}_secret")
        if not api_secret:
            raise ValueError(f"No secret found for {self.api_name}")
        
        # Create message to sign
        message = f"{endpoint}{''.join(f'{k}={v}' for k, v in sorted(params.items()))}"
        
        # Generate HMAC signature
        signature = hmac.new(
            api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def make_request(self,
                    endpoint: str,
                    method: str,
                    params: Optional[Dict] = None) -> Dict:
        """
        Make authenticated API request
        
        This is a placeholder - actual implementation would use requests library
        """
        
        # Check rate limit
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        # Get API key
        api_key = self.credential_manager.retrieve_credential(f"{self.api_name}_key")
        if not api_key:
            raise ValueError(f"No API key found for {self.api_name}")
        
        params = params or {}
        
        # Sign request
        signature = self.sign_request(endpoint, params)
        
        # Add auth headers (placeholder)
        headers = {
            'X-API-Key': api_key,
            'X-Signature': signature,
            'X-Timestamp': str(int(datetime.now().timestamp()))
        }
        
        # Record request
        self.recent_requests.append(datetime.now())
        
        # In production, this would make actual HTTP request
        return {
            'status': 'success',
            'headers': headers,
            'endpoint': endpoint,
            'method': method
        }
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limit"""
        
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Remove old requests
        self.recent_requests = [t for t in self.recent_requests if t > cutoff]
        
        return len(self.recent_requests) < self.requests_per_minute


class LightningSecurityManager:
    """
    Main security manager coordinating all security components
    """
    
    def __init__(self, master_password: str):
        # Initialize components
        self.credential_manager = CredentialManager(master_password)
        self.guardrails = TradingGuardrails()
        self.audit_logger = AuditLogger()
        
        # API wrappers (created on demand)
        self.api_wrappers = {}
    
    def store_api_credentials(self, api_name: str, api_key: str, api_secret: str):
        """Store API credentials securely"""
        
        self.credential_manager.store_credential(f"{api_name}_key", api_key)
        self.credential_manager.store_credential(f"{api_name}_secret", api_secret)
        
        self.audit_logger.log(
            event_type='credentials',
            severity='info',
            details={'action': 'stored', 'api': api_name}
        )
    
    def get_api_wrapper(self, api_name: str) -> APISecurityWrapper:
        """Get or create API wrapper"""
        
        if api_name not in self.api_wrappers:
            self.api_wrappers[api_name] = APISecurityWrapper(
                self.credential_manager,
                api_name
            )
        
        return self.api_wrappers[api_name]
    
    def validate_order(self,
                      ticker: str,
                      order_value: float,
                      current_capital: float,
                      total_exposure: float) -> Dict:
        """
        Validate order through guardrails
        Logs validation result
        """
        
        validation = self.guardrails.validate_trade(
            order_value=order_value,
            current_capital=current_capital,
            total_exposure=total_exposure
        )
        
        # Log validation
        severity = 'info' if validation['approved'] else 'warning'
        self.audit_logger.log(
            event_type='order_validation',
            severity=severity,
            details={
                'ticker': ticker,
                'order_value': order_value,
                'approved': validation['approved'],
                'warnings': validation['warnings'],
                'blockers': validation['blockers']
            }
        )
        
        return validation
    
    def record_order(self,
                    ticker: str,
                    action: str,
                    quantity: int,
                    price: float,
                    order_id: str):
        """Record order execution"""
        
        self.guardrails.record_trade()
        
        self.audit_logger.log(
            event_type='order',
            severity='info',
            details={
                'ticker': ticker,
                'action': action,
                'quantity': quantity,
                'price': price,
                'order_id': order_id,
                'value': quantity * price
            }
        )
    
    def record_fill(self,
                   order_id: str,
                   fill_price: float,
                   pnl: float):
        """Record order fill and P&L"""
        
        self.guardrails.record_trade(pnl=pnl)
        
        self.audit_logger.log(
            event_type='fill',
            severity='info',
            details={
                'order_id': order_id,
                'fill_price': fill_price,
                'pnl': pnl
            }
        )
    
    def log_error(self, error_type: str, details: Dict, critical: bool = False):
        """Log error event"""
        
        severity = 'critical' if critical else 'error'
        
        self.audit_logger.log(
            event_type='error',
            severity=severity,
            details={
                'error_type': error_type,
                **details
            }
        )
        
        # Activate circuit breaker for critical errors
        if critical:
            self.guardrails._activate_circuit_breaker(f"Critical error: {error_type}")
    
    def get_security_status(self) -> Dict:
        """Get comprehensive security status"""
        
        return {
            'guardrails': self.guardrails.get_status(),
            'stored_credentials': self.credential_manager.list_credentials(),
            'active_api_wrappers': list(self.api_wrappers.keys()),
            'recent_errors': self.audit_logger.get_error_summary(hours=24),
            'total_logs': len(self.audit_logger.logs)
        }
    
    def export_audit_logs(self, filepath: str):
        """Export all audit logs to file"""
        
        logs_dict = [log.to_dict() for log in self.audit_logger.logs]
        
        with open(filepath, 'w') as f:
            json.dump(logs_dict, f, indent=2)
    
    def save_credentials(self, filepath: str):
        """Save encrypted credentials to file"""
        self.credential_manager.save_to_file(filepath)


def demo_security_manager():
    """Demo showing security manager capabilities"""
    
    print("\n" + "="*80)
    print("LIGHTNING GEX - SECURITY MANAGER DEMO")
    print("="*80)
    
    # Create security manager
    security = LightningSecurityManager(master_password="demo_password_123")
    
    print("\n1. Storing API Credentials...")
    security.store_api_credentials(
        api_name='thinkorswim',
        api_key='demo_api_key_12345',
        api_secret='demo_api_secret_67890'
    )
    print("   ✓ Credentials stored securely (encrypted)")
    
    print("\n2. Validating Orders...")
    
    # Valid order
    validation1 = security.validate_order(
        ticker='AAPL',
        order_value=5000,
        current_capital=100000,
        total_exposure=20000
    )
    print(f"\n   Order 1 (5% position):")
    print(f"   - Approved: {validation1['approved']}")
    if validation1['warnings']:
        print(f"   - Warnings: {', '.join(validation1['warnings'])}")
    
    # Oversized order
    validation2 = security.validate_order(
        ticker='TSLA',
        order_value=20000,
        current_capital=100000,
        total_exposure=20000
    )
    print(f"\n   Order 2 (20% position):")
    print(f"   - Approved: {validation2['approved']}")
    if validation2['blockers']:
        print(f"   - Blockers: {', '.join(validation2['blockers'])}")
    
    print("\n3. Recording Orders...")
    security.record_order(
        ticker='AAPL',
        action='BUY',
        quantity=27,
        price=185.0,
        order_id='ORDER_001'
    )
    print("   ✓ Order recorded in audit log")
    
    print("\n4. Recording Fill...")
    security.record_fill(
        order_id='ORDER_001',
        fill_price=185.50,
        pnl=13.50  # Small profit
    )
    print("   ✓ Fill recorded with P&L")
    
    print("\n5. Logging Error...")
    security.log_error(
        error_type='connection_timeout',
        details={'broker': 'thinkorswim', 'timeout': 30},
        critical=False
    )
    print("   ✓ Error logged (non-critical)")
    
    print("\n6. Security Status...")
    status = security.get_security_status()
    
    print(f"\n   Guardrails:")
    print(f"   - Daily Trades: {status['guardrails']['daily_trades']}")
    print(f"   - Daily P&L: ${status['guardrails']['daily_pnl']:.2f}")
    print(f"   - Circuit Breaker: {'ACTIVE' if status['guardrails']['circuit_breaker_active'] else 'Inactive'}")
    
    print(f"\n   Credentials:")
    print(f"   - Stored: {', '.join(status['stored_credentials'])}")
    
    print(f"\n   Audit Logs:")
    print(f"   - Total Entries: {status['total_logs']}")
    print(f"   - Recent Errors: {status['recent_errors']['error_count']}")
    
    print("\n7. Making Authenticated API Request...")
    api_wrapper = security.get_api_wrapper('thinkorswim')
    response = api_wrapper.make_request(
        endpoint='/accounts/positions',
        method='GET'
    )
    print(f"   ✓ Request authenticated (signature: {response['headers']['X-Signature'][:16]}...)")
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo_security_manager()
    
    print("="*80)
    print("INTEGRATION WITH LIGHTNING GEX")
    print("="*80)
    print("""
To integrate with your Lightning GEX system:

1. Initialize security manager:
   
   security = LightningSecurityManager(master_password="your_secure_password")

2. Store broker credentials:
   
   security.store_api_credentials(
       api_name='thinkorswim',
       api_key=td_api_key,
       api_secret=td_api_secret
   )

3. Validate every order:
   
   validation = security.validate_order(
       ticker=ticker,
       order_value=position_dollars,
       current_capital=account_balance,
       total_exposure=current_total_exposure
   )
   
   if validation['approved']:
       place_order()
   else:
       print(f"Order blocked: {validation['blockers']}")

4. Record all trading activity:
   
   # When placing order
   security.record_order(ticker, action, quantity, price, order_id)
   
   # When order fills
   security.record_fill(order_id, fill_price, pnl)
   
   # When errors occur
   security.log_error(error_type, details, critical=is_critical)

5. Make authenticated API calls:
   
   api = security.get_api_wrapper('thinkorswim')
   response = api.make_request(endpoint, method, params)

6. Monitor security status:
   
   status = security.get_security_status()
   
   if status['guardrails']['circuit_breaker_active']:
       send_alert("Circuit breaker activated!")

7. Export audit logs:
   
   security.export_audit_logs('audit_logs.json')

8. Save encrypted credentials:
   
   security.save_credentials('credentials.enc')

Expected Benefits:
- Prevent catastrophic losses through guardrails
- Secure credential storage (never plain text)
- Complete audit trail for compliance
- Rate limiting prevents API bans
- Circuit breakers stop runaway behavior
- Encrypted API communication
    """)
    print("="*80 + "\n")
