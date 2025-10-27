import React from 'react'
import { 
  Users, 
  MessageSquare, 
  Clock, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle
} from 'lucide-react'

interface OverviewCardProps {
  title: string
  value: string | number
  change: string
  changeType: 'positive' | 'negative' | 'neutral'
  icon: React.ComponentType<{ className?: string }>
  color: string
}

const OverviewCard: React.FC<OverviewCardProps> = ({
  title,
  value,
  change,
  changeType,
  icon: Icon,
  color
}) => {
  const changeColor = {
    positive: 'text-green-600',
    negative: 'text-red-600',
    neutral: 'text-gray-600'
  }[changeType]
  
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          <p className={`text-sm ${changeColor} mt-1`}>
            {change}
          </p>
        </div>
        <div className={`p-3 rounded-full ${color}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  )
}

export const OverviewCards: React.FC = () => {
  const cards = [
    {
      title: 'Active Users',
      value: '1,234',
      change: '+12% from last week',
      changeType: 'positive' as const,
      icon: Users,
      color: 'bg-blue-500'
    },
    {
      title: 'Total Conversations',
      value: '45,678',
      change: '+8% from last week',
      changeType: 'positive' as const,
      icon: MessageSquare,
      color: 'bg-green-500'
    },
    {
      title: 'Avg Response Time',
      value: '1.2s',
      change: '-15% from last week',
      changeType: 'positive' as const,
      icon: Clock,
      color: 'bg-purple-500'
    },
    {
      title: 'System Health',
      value: '99.9%',
      change: 'Stable',
      changeType: 'neutral' as const,
      icon: TrendingUp,
      color: 'bg-emerald-500'
    },
    {
      title: 'Active Sessions',
      value: '89',
      change: '+5% from last hour',
      changeType: 'positive' as const,
      icon: Users,
      color: 'bg-orange-500'
    },
    {
      title: 'Error Rate',
      value: '0.1%',
      change: '-0.2% from last week',
      changeType: 'positive' as const,
      icon: AlertTriangle,
      color: 'bg-red-500'
    }
  ]
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {cards.map((card, index) => (
        <OverviewCard key={index} {...card} />
      ))}
    </div>
  )
}


