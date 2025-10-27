"use client"

import type React from "react"

import { createContext, useContext, useState, useEffect } from "react"

type UserRole = "physician" | "user" | null

interface AuthContextType {
  isAuthenticated: boolean
  userRole: UserRole
  login: (email: string, password: string, role: "physician" | "user") => Promise<boolean>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [userRole, setUserRole] = useState<UserRole>(null)

  useEffect(() => {
    // Check if user is already logged in
    const storedAuth = localStorage.getItem("auth")
    if (storedAuth) {
      const { role } = JSON.parse(storedAuth)
      setIsAuthenticated(true)
      setUserRole(role)
    }
  }, [])

  const login = async (email: string, password: string, role: "physician" | "user") => {
    // Mock authentication - replace with real API call
    if (email && password) {
      setIsAuthenticated(true)
      setUserRole(role)
      localStorage.setItem("auth", JSON.stringify({ role, email }))
      return true
    }
    return false
  }

  const logout = () => {
    setIsAuthenticated(false)
    setUserRole(null)
    localStorage.removeItem("auth")
  }

  return <AuthContext.Provider value={{ isAuthenticated, userRole, login, logout }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
