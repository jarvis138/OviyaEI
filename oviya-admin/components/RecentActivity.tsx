import React from 'react'
import { formatDistanceToNow } from 'date-fns'
import { MessageSquare, User, AlertTriangle, CheckCircle, Clock } from 'lucide-react'

interface ActivityItem {
  id: string
  type: 'conversation' | 'user' | 'error' | 'system'
  title: string
  description: string
  timestamp: Date
  status?: 'success' | 'warning' | 'error'
}

const ActivityItem: React.FC<{ item: ActivityItem }> = ({ item }) => {
  const getIcon = () => {
    switch (item.type) {
      case 'conversation':
        return MessageSquare
      case 'user':
        return User
      case 'error':
        return AlertTriangle
      case 'system':
        return CheckCircle
      default:
        return Clock
    }
  }
  
  const getStatusColor = () => {
    switch (item.status) {
      case 'success':
        return 'text-green-600'
      case 'warning':
        return 'text-yellow-600'
      case 'error':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }
  
  const Icon = getIcon()
  
  return (
    <div className="flex items-start space-x-3 p-4 hover:bg-gray-50 rounded-lg transition-colors">
      <div className={`p-2 rounded-full bg-gray-100 ${getStatusColor()}`}>
        <Icon className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-gray-900">{item.title}</p>
          <p className="text-xs text-gray-500">
            {formatDistanceToNow(item.timestamp, { addSuffix: true })}
          </p>
        </div>
        <p className="text-sm text-gray-600 mt-1">{item.description}</p>
      </div>
    </div>
  )
}

export const RecentActivity: React.FC = () => {
  const activities: ActivityItem[] = [
    {
      id: '1',
      type: 'conversation',
      title: 'New conversation started',
      description: 'User john_doe started a conversation with empathetic emotion',
      timestamp: new Date(Date.now() - 2 * 60 * 1000), // 2 minutes ago
      status: 'success'
    },
    {
      id: '2',
      type: 'user',
      title: 'User registered',
      description: 'New user sarah_wilson registered with email verification',
      timestamp: new Date(Date.now() - 5 * 60 * 1000), // 5 minutes ago
      status: 'success'
    },
    {
      id: '3',
      type: 'error',
      title: 'CSM Service timeout',
      description: 'CSM service experienced timeout for session abc123',
      timestamp: new Date(Date.now() - 8 * 60 * 1000), // 8 minutes ago
      status: 'warning'
    },
    {
      id: '4',
      type: 'system',
      title: 'System backup completed',
      description: 'Daily backup completed successfully',
      timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
      status: 'success'
    },
    {
      id: '5',
      type: 'conversation',
      title: 'Conversation ended',
      description: 'User mike_chen ended conversation after 5 minutes',
      timestamp: new Date(Date.now() - 20 * 60 * 1000), // 20 minutes ago
      status: 'success'
    },
    {
      id: '6',
      type: 'error',
      title: 'Rate limit exceeded',
      description: 'User exceeded rate limit for API calls',
      timestamp: new Date(Date.now() - 25 * 60 * 1000), // 25 minutes ago
      status: 'error'
    },
    {
      id: '7',
      type: 'system',
      title: 'Service health check',
      description: 'All services passed health check',
      timestamp: new Date(Date.now() - 30 * 60 * 1000), // 30 minutes ago
      status: 'success'
    },
    {
      id: '8',
      type: 'user',
      title: 'User logged out',
      description: 'User emma_davis logged out from session',
      timestamp: new Date(Date.now() - 35 * 60 * 1000), // 35 minutes ago
      status: 'success'
    }
  ]
  
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
        <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">
          View All
        </button>
      </div>
      
      <div className="space-y-1">
        {activities.map((activity) => (
          <ActivityItem key={activity.id} item={activity} />
        ))}
      </div>
      
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-gray-600">Success</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <span className="text-gray-600">Warning</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              <span className="text-gray-600">Error</span>
            </div>
          </div>
          <button className="text-blue-600 hover:text-blue-700 font-medium">
            Export Logs
          </button>
        </div>
      </div>
    </div>
  )
}


