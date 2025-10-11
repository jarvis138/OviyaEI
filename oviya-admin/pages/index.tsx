import { NextPage } from 'next'
import Head from 'next/head'
import { DashboardLayout } from '@/components/DashboardLayout'
import { OverviewCards } from '@/components/OverviewCards'
import { SystemMetrics } from '@/components/SystemMetrics'
import { RecentActivity } from '@/components/RecentActivity'
import { ServiceStatus } from '@/components/ServiceStatus'

const Dashboard: NextPage = () => {
  return (
    <>
      <Head>
        <title>Oviya AI Admin Dashboard</title>
        <meta name="description" content="Admin dashboard for Oviya AI system monitoring and management" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <DashboardLayout>
        <div className="space-y-6">
          {/* Page Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
              <p className="text-gray-600 mt-1">
                Monitor and manage your Oviya AI system
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500">
                Last updated: {new Date().toLocaleTimeString()}
              </div>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                Refresh Data
              </button>
            </div>
          </div>
          
          {/* Overview Cards */}
          <OverviewCards />
          
          {/* System Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SystemMetrics />
            <ServiceStatus />
          </div>
          
          {/* Recent Activity */}
          <RecentActivity />
        </div>
      </DashboardLayout>
    </>
  )
}

export default Dashboard


