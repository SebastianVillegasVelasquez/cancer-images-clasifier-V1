"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useAuth } from "@/lib/auth-context"

export default function LoginPage() {
  const [physicianEmail, setPhysicianEmail] = useState("")
  const [physicianPassword, setPhysicianPassword] = useState("")
  const [physicianId, setPhysicianId] = useState("")

  const [userEmail, setUserEmail] = useState("")
  const [userPassword, setUserPassword] = useState("")
  const [userId, setUserId] = useState("")

  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const { login } = useAuth()
  const router = useRouter()

  const handlePhysicianLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    if (!physicianEmail || !physicianPassword || !physicianId) {
      setError("Please fill in all fields")
      setLoading(false)
      return
    }

    const success = await login(physicianEmail, physicianPassword, "physician")

    if (success) {
      router.push("/physician/dashboard")
    } else {
      setError("Invalid credentials")
    }

    setLoading(false)
  }

  const handleUserLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    if (!userEmail || !userPassword || !userId) {
      setError("Please fill in all fields")
      setLoading(false)
      return
    }

    const success = await login(userEmail, userPassword, "user")

    if (success) {
      router.push("/user/dashboard")
    } else {
      setError("Invalid credentials")
    }

    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-secondary p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center gap-2 mb-4">
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
          <h1 className="text-2xl font-bold">Welcome Back</h1>
          <p className="text-muted-foreground mt-2">Sign in to access your account</p>
        </div>

        <Tabs defaultValue="physician" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="physician">Physician</TabsTrigger>
            <TabsTrigger value="user">Patient</TabsTrigger>
          </TabsList>

          <TabsContent value="physician">
            <Card>
              <CardHeader>
                <CardTitle>Physician Login</CardTitle>
                <CardDescription>Enter your credentials and medical license ID</CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handlePhysicianLogin} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="physician-id">Medical License ID</Label>
                    <Input
                      id="physician-id"
                      placeholder="e.g., MD123456"
                      value={physicianId}
                      onChange={(e) => setPhysicianId(e.target.value)}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="physician-email">Email</Label>
                    <Input
                      id="physician-email"
                      type="email"
                      placeholder="doctor@example.com"
                      value={physicianEmail}
                      onChange={(e) => setPhysicianEmail(e.target.value)}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="physician-password">Password</Label>
                    <Input
                      id="physician-password"
                      type="password"
                      value={physicianPassword}
                      onChange={(e) => setPhysicianPassword(e.target.value)}
                      required
                    />
                  </div>

                  {error && <p className="text-sm text-red-600">{error}</p>}

                  <Button type="submit" className="w-full" disabled={loading}>
                    {loading ? "Signing in..." : "Sign In as Physician"}
                  </Button>

                  <div className="text-sm text-center text-muted-foreground">
                    <Link href="#" className="hover:text-foreground transition-colors">
                      Forgot password?
                    </Link>
                  </div>
                </form>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="user">
            <Card>
              <CardHeader>
                <CardTitle>Patient Login</CardTitle>
                <CardDescription>Enter your credentials and patient ID</CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleUserLogin} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="user-id">Patient ID</Label>
                    <Input
                      id="user-id"
                      placeholder="e.g., PT789012"
                      value={userId}
                      onChange={(e) => setUserId(e.target.value)}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="user-email">Email</Label>
                    <Input
                      id="user-email"
                      type="email"
                      placeholder="patient@example.com"
                      value={userEmail}
                      onChange={(e) => setUserEmail(e.target.value)}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="user-password">Password</Label>
                    <Input
                      id="user-password"
                      type="password"
                      value={userPassword}
                      onChange={(e) => setUserPassword(e.target.value)}
                      required
                    />
                  </div>

                  {error && <p className="text-sm text-red-600">{error}</p>}

                  <Button type="submit" className="w-full" disabled={loading}>
                    {loading ? "Signing in..." : "Sign In as Patient"}
                  </Button>

                  <div className="text-sm text-center text-muted-foreground">
                    <Link href="#" className="hover:text-foreground transition-colors">
                      Forgot password?
                    </Link>
                  </div>
                </form>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <p className="text-center text-sm text-muted-foreground mt-6">
          Don't have an account?{" "}
          <Link href="#" className="text-primary hover:underline">
            Contact your healthcare provider
          </Link>
        </p>
      </div>
    </div>
  )
}
