import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import { DashboardPage } from './pages/DashboardPage'
import { UploadVideoPage } from './pages/UploadVideoPage'
import { LiveStreamPage } from './pages/LiveStreamPage'
import { AlertsHistoryPage } from './pages/AlertsHistoryPage'
import { SettingsPage } from './pages/SettingsPage'

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadVideoPage />} />
        <Route path="/live" element={<LiveStreamPage />} />
        <Route path="/alerts" element={<AlertsHistoryPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
