import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'

const data = [
  { time: '00:00', latency: 1.2, throughput: 120, errors: 2 },
  { time: '01:00', latency: 1.1, throughput: 135, errors: 1 },
  { time: '02:00', latency: 1.0, throughput: 98, errors: 0 },
  { time: '03:00', latency: 1.3, throughput: 156, errors: 3 },
  { time: '04:00', latency: 1.1, throughput: 142, errors: 1 },
  { time: '05:00', latency: 1.4, throughput: 178, errors: 2 },
  { time: '06:00', latency: 1.2, throughput: 201, errors: 1 },
  { time: '07:00', latency: 1.5, throughput: 234, errors: 4 },
  { time: '08:00', latency: 1.8, throughput: 267, errors: 3 },
  { time: '09:00', latency: 1.6, throughput: 289, errors: 2 },
  { time: '10:00', latency: 1.4, throughput: 312, errors: 1 },
  { time: '11:00', latency: 1.7, throughput: 298, errors: 3 },
]

export const SystemMetrics: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">System Metrics</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span className="text-sm text-gray-600">Live</span>
        </div>
      </div>
      
      <div className="space-y-6">
        {/* Latency Chart */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Response Latency (seconds)</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="time" 
                stroke="#666"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                stroke="#666"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                domain={[0, 2]}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="latency" 
                stroke="#8b5cf6" 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, stroke: '#8b5cf6', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Throughput Chart */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Requests per Minute</h4>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="time" 
                stroke="#666"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                stroke="#666"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="throughput" 
                stroke="#10b981" 
                fill="#10b981"
                fillOpacity={0.1}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        
        {/* Error Rate */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Error Rate</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="time" 
                stroke="#666"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                stroke="#666"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                domain={[0, 5]}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="errors" 
                stroke="#ef4444" 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, stroke: '#ef4444', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}


