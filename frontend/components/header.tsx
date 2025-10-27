"use client"

import Link from "next/link"
import { useAuth } from "@/lib/auth-context"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"

export function Header() {
  const { isAuthenticated, userRole, logout } = useAuth()
  const router = useRouter()

  const handleLogout = () => {
    logout()
    router.push("/")
  }

  return (
    <header className="border-b bg-background sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
            <svg className="w-6 h-6 text-primary-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
              />
            </svg>
          </div>
          <span className="text-xl font-semibold">MedConnect</span>
        </Link>

        <nav className="hidden md:flex items-center gap-6">
          <Link href="/#about" className="text-muted-foreground hover:text-foreground transition-colors">
            About
          </Link>
          <Link href="/#services" className="text-muted-foreground hover:text-foreground transition-colors">
            Services
          </Link>
          <Link href="/#contact" className="text-muted-foreground hover:text-foreground transition-colors">
            Contact
          </Link>
        </nav>

        <div className="flex items-center gap-3">
          {isAuthenticated ? (
            <>
              <Link href={userRole === "physician" ? "/physician/dashboard" : "/user/dashboard"}>
                <Button variant="outline">Dashboard</Button>
              </Link>
              <Button onClick={handleLogout} variant="outline">
                Logout
              </Button>
            </>
          ) : (
            <Link href="/login">
              <Button>Login</Button>
            </Link>
          )}
        </div>
      </div>
    </header>
  )
}
