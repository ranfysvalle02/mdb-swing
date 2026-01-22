"""Alpaca Account Management Service.

Manages multiple Alpaca paper trading accounts, allowing users to:
- Store multiple account credentials
- Select which account to use for trading
- Switch between accounts dynamically
- Create/manage account configurations

MDB-Engine Integration:
- Logging: Uses `get_logger(__name__)` from mdb_engine.observability
- Database: Uses scoped database via mdb-engine
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from mdb_engine.observability import get_logger
import alpaca_trade_api as tradeapi
import hashlib

logger = get_logger(__name__)


class AlpacaAccountManager:
    """Manages multiple Alpaca paper trading accounts."""
    
    def __init__(self, db):
        """Initialize account manager with database connection.
        
        Args:
            db: MongoDB database instance (scoped via mdb-engine)
        """
        self.db = db
    
    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Get all stored Alpaca accounts.
        
        Returns:
            List of account dictionaries (without secrets)
        """
        try:
            accounts = await self.db.alpaca_accounts.find({}).to_list(length=100)
            # Remove sensitive data before returning
            for account in accounts:
                account.pop('api_secret', None)
                account.pop('_id', None)
            return accounts
        except Exception as e:
            logger.error(f"Failed to get accounts: {e}", exc_info=True)
            return []
    
    async def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific account by ID.
        
        Args:
            account_id: Account identifier
            
        Returns:
            Account dictionary with all fields (including secrets for internal use)
        """
        try:
            account = await self.db.alpaca_accounts.find_one({"account_id": account_id})
            if account:
                account.pop('_id', None)
            return account
        except Exception as e:
            logger.error(f"Failed to get account {account_id}: {e}", exc_info=True)
            return None
    
    async def get_active_account(self) -> Optional[Dict[str, Any]]:
        """Get the currently active account.
        
        Returns:
            Active account dictionary or None
        """
        try:
            account = await self.db.alpaca_accounts.find_one({"is_active": True})
            if account:
                account.pop('_id', None)
            return account
        except Exception as e:
            logger.error(f"Failed to get active account: {e}", exc_info=True)
            return None
    
    async def create_account(
        self,
        account_id: str,
        nickname: str,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets",
        is_active: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Create a new Alpaca account configuration.
        
        Args:
            account_id: Unique account identifier (e.g., "PA3AFGM5YBAO")
            nickname: User-friendly name for the account
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Alpaca API base URL (default: paper trading)
            is_active: Whether this account should be active
            
        Returns:
            Created account dictionary or None if failed
        """
        try:
            # Validate credentials by testing connection
            test_api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            try:
                test_account = test_api.get_account()
                account_number = test_account.account_number
            except Exception as e:
                logger.error(f"Failed to validate Alpaca credentials: {e}")
                return None
            
            # If this account should be active, deactivate others
            if is_active:
                await self.db.alpaca_accounts.update_many(
                    {"is_active": True},
                    {"$set": {"is_active": False, "updated_at": datetime.now()}}
                )
            
            # Create account document
            account_doc = {
                "account_id": account_id,
                "account_number": account_number,
                "nickname": nickname,
                "api_key": api_key,
                "api_secret": api_secret,  # Encrypted in production
                "base_url": base_url,
                "is_active": is_active,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Check if account already exists
            existing = await self.db.alpaca_accounts.find_one({"account_id": account_id})
            if existing:
                # Update existing account
                await self.db.alpaca_accounts.update_one(
                    {"account_id": account_id},
                    {"$set": {
                        "nickname": nickname,
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "base_url": base_url,
                        "is_active": is_active,
                        "updated_at": datetime.now()
                    }}
                )
                account_doc.pop('created_at')
            else:
                # Insert new account
                await self.db.alpaca_accounts.insert_one(account_doc)
            
            # Return account without secret
            account_doc.pop('api_secret', None)
            account_doc.pop('_id', None)
            return account_doc
            
        except Exception as e:
            logger.error(f"Failed to create account {account_id}: {e}", exc_info=True)
            return None
    
    async def set_active_account(self, account_id: str) -> bool:
        """Set an account as active (deactivates others).
        
        Args:
            account_id: Account identifier to activate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if account exists
            account = await self.db.alpaca_accounts.find_one({"account_id": account_id})
            if not account:
                logger.warning(f"Account {account_id} not found")
                return False
            
            # Deactivate all accounts
            await self.db.alpaca_accounts.update_many(
                {},
                {"$set": {"is_active": False, "updated_at": datetime.now()}}
            )
            
            # Activate selected account
            await self.db.alpaca_accounts.update_one(
                {"account_id": account_id},
                {"$set": {"is_active": True, "updated_at": datetime.now()}}
            )
            
            logger.info(f"Activated account {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set active account {account_id}: {e}", exc_info=True)
            return False
    
    async def delete_account(self, account_id: str) -> bool:
        """Delete an account configuration.
        
        Args:
            account_id: Account identifier to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.db.alpaca_accounts.delete_one({"account_id": account_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted account {account_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete account {account_id}: {e}", exc_info=True)
            return False
    
    def get_api_client(self, account: Optional[Dict[str, Any]] = None) -> Optional[tradeapi.REST]:
        """Get Alpaca API client for an account.
        
        Args:
            account: Account dictionary. If None, uses active account.
            
        Returns:
            Alpaca REST API client or None
        """
        try:
            if account is None:
                # This is a sync method, so we can't await here
                # Caller should pass account explicitly
                return None
            
            if not account.get('api_key') or not account.get('api_secret'):
                logger.warning("Account missing API credentials")
                return None
            
            return tradeapi.REST(
                account['api_key'],
                account['api_secret'],
                account.get('base_url', 'https://paper-api.alpaca.markets'),
                api_version='v2'
            )
        except Exception as e:
            logger.error(f"Failed to create API client: {e}", exc_info=True)
            return None
    
    async def get_api_client_for_active(self) -> Optional[tradeapi.REST]:
        """Get API client for the active account.
        
        Returns:
            Alpaca REST API client or None
        """
        account = await self.get_active_account()
        if not account:
            return None
        return self.get_api_client(account)
    
    async def test_connection(self, account_id: str) -> Dict[str, Any]:
        """Test connection to an Alpaca account.
        
        Args:
            account_id: Account identifier to test
            
        Returns:
            Dictionary with test results
        """
        try:
            account = await self.get_account(account_id)
            if not account:
                return {"success": False, "error": "Account not found"}
            
            api = self.get_api_client(account)
            if not api:
                return {"success": False, "error": "Failed to create API client"}
            
            # Test connection
            acct = api.get_account()
            return {
                "success": True,
                "account_number": acct.account_number,
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "status": acct.status
            }
        except Exception as e:
            logger.error(f"Connection test failed for {account_id}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
