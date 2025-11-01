#!/usr/bin/env python3
"""
MCP Client Wrapper for Oviya
Real MCP client implementation supporting stdio and SSE connections
"""

import asyncio
import json
import subprocess
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False
    logger.warning("MCP SDK not available, using fallback mock client")


class MCPClient:
    """
    Real MCP client wrapper supporting stdio and SSE connections
    
    CSM-1B Compatible: All tool calls return data that can be used for CSM-1B
    context enhancement, prosody modulation, and therapeutic response generation
    """
    
    def __init__(self, server_name: str, config: Optional[Dict] = None):
        """
        Initialize MCP client
        
        Args:
            server_name: Name of MCP server (from config)
            config: MCP server configuration dict with command, args, env
        """
        self.server_name = server_name
        self.config = config or {}
        self.session: Optional[ClientSession] = None
        self.read_stream = None
        self.write_stream = None
        self.connection_type = None  # 'stdio' or 'sse'
        self._initialized = False
    
    async def initialize(self):
        """Initialize connection to MCP server"""
        if self._initialized:
            return
        
        if not MCP_SDK_AVAILABLE:
            logger.warning(f"MCP SDK not available for {self.server_name}, using mock")
            self._initialized = True
            return
        
        try:
            # Determine connection type from config
            command = self.config.get("command", "")
            
            # For stdio connections, we need to keep the context manager alive
            # Store the context manager to keep connection alive
            if command in ["npx", "node", "python", "bun", "bunx"]:
                self.connection_type = "stdio"
                args = self.config.get("args", [])
                env = self.config.get("env", {})
                
                # Create stdio client context manager
                self._stdio_context = stdio_client(
                    command=command,
                    args=args,
                    env={**os.environ, **env}
                )
                self._read_write_streams = await self._stdio_context.__aenter__()
                self.read_stream, self.write_stream = self._read_write_streams
                self.session = ClientSession(self.read_stream, self.write_stream)
                await self.session.initialize()
                self._initialized = True
                logger.info(f"MCP client initialized: {self.server_name} (stdio)")
            
            elif "http" in command.lower() or self.config.get("url"):
                # Use SSE for HTTP endpoints
                self.connection_type = "sse"
                url = command if command.startswith("http") else self.config.get("url", "")
                
                # Create SSE client context manager
                self._sse_context = sse_client(url=url, timeout=20)
                self._read_write_streams = await self._sse_context.__aenter__()
                self.read_stream, self.write_stream = self._read_write_streams
                self.session = ClientSession(self.read_stream, self.write_stream)
                await self.session.initialize()
                self._initialized = True
                logger.info(f"MCP client initialized: {self.server_name} (sse)")
            
            else:
                # Try stdio for other commands
                self.connection_type = "stdio"
                args = self.config.get("args", [])
                env = self.config.get("env", {})
                
                self._stdio_context = stdio_client(
                    command=command,
                    args=args,
                    env={**os.environ, **env}
                )
                self._read_write_streams = await self._stdio_context.__aenter__()
                self.read_stream, self.write_stream = self._read_write_streams
                self.session = ClientSession(self.read_stream, self.write_stream)
                await self.session.initialize()
                self._initialized = True
                logger.info(f"MCP client initialized: {self.server_name} (stdio)")
        
        except Exception as e:
            logger.error(f"Failed to initialize MCP client {self.server_name}: {e}")
            self._initialized = False
            # Don't raise - fallback to mock
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call MCP tool
        
        CSM-1B Compatible: Returns data that can enhance:
        - Conversation context
        - Prosody parameters
        - Therapeutic response generation
        
        Args:
            tool_name: Name of tool to call
            params: Tool parameters
            
        Returns:
            Tool result as dict
        """
        if not self._initialized:
            await self.initialize()
        
        if not MCP_SDK_AVAILABLE or not self.session:
            # Fallback to mock
            return await self._mock_call_tool(tool_name, params)
        
        try:
            # Call tool via MCP session
            result_list = await self.session.call_tool(tool_name, params)
            
            # Parse result (MCP returns list of TextContent)
            if result_list and isinstance(result_list, list) and len(result_list) > 0:
                content_item = result_list[0]
                if hasattr(content_item, 'type') and content_item.type == "text":
                    try:
                        # Try to parse as JSON
                        data = json.loads(content_item.text)
                        return data
                    except json.JSONDecodeError:
                        # Return as text
                        return {"text": content_item.text, "raw": True}
                elif hasattr(content_item, 'text'):
                    return {"text": content_item.text, "raw": True}
            
            return {"result": "empty", "tool": tool_name}
        
        except Exception as e:
            logger.error(f"MCP tool call failed {self.server_name}.{tool_name}: {e}")
            # Fallback to mock
            return await self._mock_call_tool(tool_name, params)
    
    async def _mock_call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback mock tool call"""
        logger.warning(f"Using mock tool call for {self.server_name}.{tool_name}")
        return {"mock": True, "tool": tool_name, "params": params}
    
    async def close(self):
        """Close MCP connection"""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Error closing MCP session: {e}")
        
        # Exit context managers
        if hasattr(self, '_stdio_context') and self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except Exception:
                pass
        
        if hasattr(self, '_sse_context') and self._sse_context:
            try:
                await self._sse_context.__aexit__(None, None, None)
            except Exception:
                pass
        
        self._initialized = False


class MCPClientManager:
    """
    Manager for multiple MCP clients
    
    Loads configuration and initializes clients on demand
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize MCP client manager
        
        Args:
            config_path: Path to MCP config JSON file
        """
        self.config_path = config_path or Path("mcp-ecosystem/config/oviya-mcp-config.json")
        self.config: Dict[str, Dict] = {}
        self.clients: Dict[str, MCPClient] = {}
        self._load_config()
    
    def _load_config(self):
        """Load MCP configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    self.config = config_data.get("mcpServers", {})
                    logger.info(f"Loaded MCP config: {len(self.config)} servers")
            else:
                logger.warning(f"MCP config not found: {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
    
    def get_client(self, server_name: str) -> Optional[MCPClient]:
        """
        Get or create MCP client for server
        
        Args:
            server_name: Name of MCP server
            
        Returns:
            MCPClient instance or None
        """
        if server_name in self.clients:
            return self.clients[server_name]
        
        if server_name not in self.config:
            logger.warning(f"MCP server not in config: {server_name}")
            return None
        
        server_config = self.config[server_name]
        client = MCPClient(server_name, server_config)
        self.clients[server_name] = client
        return client
    
    async def initialize_all(self):
        """Initialize all configured MCP clients"""
        for server_name in self.config.keys():
            try:
                client = self.get_client(server_name)
                if client:
                    await client.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize {server_name}: {e}")


# Global MCP client manager
_mcp_manager: Optional[MCPClientManager] = None

def get_mcp_client(server_name: str) -> Optional[MCPClient]:
    """Get MCP client for server (singleton)"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
    return _mcp_manager.get_client(server_name)

async def initialize_mcp_clients():
    """Initialize all MCP clients"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
    await _mcp_manager.initialize_all()

