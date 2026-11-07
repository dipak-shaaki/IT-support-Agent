import { ChatPage } from './pages/ChatPage'
import { DashboardPage } from './pages/DashboardPage'

export default function App() {
  return window.location.pathname.startsWith('/dashboard')
    ? <DashboardPage />
    : <ChatPage />
}
