TASKS = {
    "easy_sql_injection": {
        "difficulty": "easy",
        "description": (
            "Review this authentication module for SQL injection vulnerabilities. "
            "Not every f-string or string operation is vulnerable — look carefully "
            "at which ones touch SQL queries vs logging."
        ),
        "code": '''import sqlite3
import hashlib

def get_user_by_id(conn, user_id):
    """Fetch a user record by primary key."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()

def log_attempt(username, success):
    """Append an authentication event to the audit log."""
    status = "SUCCESS" if success else "FAILURE"
    print(f"[AUDIT] login {status} for user: {username}")

def login(conn, username, password):
    """Authenticate a user against the database."""
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    query = f"SELECT id FROM users WHERE username=\'{username}\' AND pw_hash=\'{pw_hash}\'"
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    if result:
        log_attempt(username, True)
    else:
        log_attempt(username, False)
    return result is not None

def get_users_by_role(conn, role):
    """Return all accounts with the specified access role."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username FROM users WHERE role = ?", (role,)
    )
    return cursor.fetchall()

def render_welcome(username):
    """Generate a personalized welcome message for the UI."""
    return f"Welcome back, {username}!"
''',
        "issues": [
            {
                "type": "security",
                "line": 18,
                "description": "SQL injection: username and pw_hash interpolated directly into query via f-string; attacker controls username to bypass authentication",
                "severity": "critical"
            }
        ]
    },

    "medium_race_condition": {
        "difficulty": "medium",
        "description": (
            "Find the concurrency bugs in this BankAccount class. "
            "Some methods are correctly protected — focus on the ones that are not."
        ),
        "code": '''import threading
import time

class BankAccount:
    def __init__(self, owner):
        self.owner = owner
        self.balance = 0.0
        self._lock = threading.Lock()

    def get_balance(self):
        """Return the current account balance."""
        return self.balance

    def deposit(self, amount):
        """Credit the account with the specified amount."""
        current = self.balance
        time.sleep(0.001)  # Simulate network confirmation delay
        self.balance = current + amount

    def withdraw(self, amount):
        """Remove funds if sufficient balance exists."""
        if self.balance >= amount:
            current = self.balance
            time.sleep(0.001)  # Simulate authorization call
            self.balance = current - amount
            return True
        return False

    def transfer(self, target, amount):
        """Move funds atomically — acquires both locks in consistent order."""
        for acct in sorted([self, target], key=id):
            acct._lock.acquire()
        try:
            if self.balance >= amount:
                self.balance -= amount
                target.balance += amount
                return True
            return False
        finally:
            for acct in sorted([self, target], key=id):
                acct._lock.release()

    def get_statement(self, currency="USD"):
        """Return a formatted account statement string."""
        with self._lock:
            return f"{self.owner}: {self.balance:.2f} {currency}"

    def freeze(self):
        """Mark the account as frozen, blocking further withdrawals."""
        with self._lock:
            self._frozen = True
''',
        "issues": [
            {
                "type": "bug",
                "line": 16,
                "description": "Race condition in deposit: read at line 16 and write at line 18 are not atomic; concurrent deposits lose updates",
                "severity": "high"
            },
            {
                "type": "bug",
                "line": 22,
                "description": "TOCTOU race in withdraw: balance checked at line 22 without a lock; another thread can withdraw between the check and the debit, causing overdraft",
                "severity": "high"
            }
        ]
    },

    "hard_memory_leak": {
        "difficulty": "hard",
        "description": (
            "Find all bugs and performance problems in this TTL cache manager. "
            "Some iteration patterns here are intentionally safe — "
            "identify only the genuinely broken ones."
        ),
        "code": '''import time

class CacheManager:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.listeners = []
        self.max_size = max_size

    def register_listener(self, callback):
        """Register a callback invoked on every cache write."""
        self.listeners.append(callback)

    def set(self, key, value, ttl=None):
        """Store a value with optional TTL (seconds)."""
        self.cache[key] = {
            "value": value,
            "expires": time.time() + ttl if ttl else None,
        }
        for listener in self.listeners:
            listener("set", key, value)

    def get(self, key):
        """Retrieve a value, returning None if missing or expired."""
        entry = self.cache.get(key)
        if entry is None:
            return None
        if entry["expires"] and time.time() > entry["expires"]:
            return None
        return entry["value"]

    def get_stats(self):
        """Return aggregate statistics for the cache."""
        total = len(self.cache)
        expired = sum(
            1 for e in self.cache.values()
            if e["expires"] and time.time() > e["expires"]
        )
        return {"total": total, "expired": expired, "active": total - expired}

    def cleanup_expired(self):
        """Remove all expired entries from the cache."""
        for key in self.cache.keys():
            entry = self.cache[key]
            if entry["expires"] and time.time() > entry["expires"]:
                del self.cache[key]

    def invalidate_prefix(self, prefix):
        """Evict all entries whose key starts with the given prefix."""
        to_delete = [k for k in self.cache if k.startswith(prefix)]
        for key in to_delete:
            del self.cache[key]

    def get_active_values(self):
        """Return a snapshot list of all non-expired cached values."""
        now = time.time()
        return [
            e["value"]
            for e in self.cache.values()
            if not e["expires"] or now <= e["expires"]
        ]
''',
        "issues": [
            {
                "type": "performance",
                "line": 11,
                "description": "Memory leak: register_listener() only appends callbacks, never removes them; listeners accumulate indefinitely in long-running processes",
                "severity": "high"
            },
            {
                "type": "performance",
                "line": 27,
                "description": "Cache bloat: expiry detected at line 27 in get() but entry is returned as None without del self.cache[key]; stale data accumulates consuming memory",
                "severity": "medium"
            },
            {
                "type": "bug",
                "line": 42,
                "description": "RuntimeError: dictionary changed size during iteration — cleanup_expired() calls del self.cache[key] while iterating self.cache.keys(); fix with list(self.cache.keys())",
                "severity": "high"
            }
        ]
    }
}
