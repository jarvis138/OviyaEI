import React from 'react'
import { CheckCircle, XCircle, AlertTriangle, Clock, Server, Database, Cpu, HardDrive } from 'lucide-react'

interface ServiceStatusProps {
  service: string
  status: 'healthy' | 'degraded' | 'down'
  uptime: string
  responseTime: string
  lastCheck: string
}

const ServiceCard: React.FC<ServiceStatusProps> = ({
  service,
  status,
  uptime,
  responseTime,
  lastCheck
}) => {
  const statusConfig = {
    healthy: {
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      label: 'Healthy'
    },
    degraded: {
      icon: AlertTriangle,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-100',
      label: 'Degraded'
    },
    down: {
      icon: XCircle,
      color: 'text-red-600',
      bgColor: 'bg-red-100',
      label: 'Down'
    }
  }
  
  const config = statusConfig[status]
  const Icon = config.icon
  
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-full ${config.bgColor}`}>
            <Icon className={`w-4 h-4 ${config.color}`} />
          </div>
          <div>
            <h4 className="font-medium text-gray-900">{service}</h4>
            <p className={`text-sm ${config.color}`}>{config.label}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-600">Uptime</p>
          <p className="font-medium text-gray-900">{uptime}</p>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-gray-600">Response Time</p>
          <p className="font-medium text-gray-900">{responseTime}</p>
        </div>
        <div>
          <p className="text-gray-600">Last Check</p>
          <p className="font-medium text-gray-900">{lastCheck}</p>
        </div>
      </div>
    </div>
  )
}

export const ServiceStatus: React.FC = () => {
  const services = [
    {
      service: 'CSM Service',
      status: 'healthy' as const,
      uptime: '99.9%',
      responseTime: '450ms',
      lastCheck: '2 min ago'
    },
    {
      service: 'ASR Service',
      status: 'healthy' as const,
      uptime: '99.8%',
      responseTime: '320ms',
      lastCheck: '1 min ago'
    },
    {
      service: 'Orchestrator',
      status: 'healthy' as const,
      uptime: '99.9%',
      responseTime: '1.2s',
      lastCheck: '30 sec ago'
    },
    {
      service: 'Database',
      status: 'healthy' as const,
      uptime: '99.9%',
      responseTime: '15ms',
      lastCheck: '1 min ago'
    },
    {
      service: 'Redis Cache',
      status: 'degraded' as const,
      uptime: '98.5%',
      responseTime: '45ms',
      lastCheck: '2 min ago'
    },
    {
      service: 'Load Balancer',
      status: 'healthy' as const,
      uptime: '99.9%',
      responseTime: '8ms',
      lastCheck: '30 sec ago'
    }
  ]
  
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Service Status</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span className="text-sm text-gray-600">All Systems Operational</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 gap-4">
        {services.map((service, index) => (
          <ServiceCard key={index} {...service} />
        ))}
      </div>
      
      {/* System Resources */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-700 mb-4">System Resources</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center space-x-3">
            <Cpu className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-sm text-gray-600">CPU Usage</p>
              <p className="font-medium text-gray-900">45%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <HardDrive className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-sm text-gray-600">Memory Usage</p>
              <p className="font-medium text-gray-900">67%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Database className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-sm text-gray-600">Disk Usage</p>
              <p className="font-medium text-gray-900">23%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Server className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-sm text-gray-600">Network I/O</p>
              <p className="font-medium text-gray-900">12 MB/s</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}


