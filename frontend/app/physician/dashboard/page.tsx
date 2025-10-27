"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"
import { Header } from "@/components/header"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function PhysicianDashboard() {
  const { isAuthenticated, userRole } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isAuthenticated || userRole !== "physician") {
      router.push("/login")
    }
  }, [isAuthenticated, userRole, router])

  if (!isAuthenticated || userRole !== "physician") {
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Physician Dashboard</h1>
          <p className="text-muted-foreground">Welcome back, Dr. Smith</p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                Today's Appointments
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">12</p>
              <p className="text-sm text-muted-foreground mt-1">3 pending confirmations</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <svg className="w-5 h-5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                  />
                </svg>
                Active Patients
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">248</p>
              <p className="text-sm text-muted-foreground mt-1">15 new this month</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
                Unread Messages
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">7</p>
              <p className="text-sm text-muted-foreground mt-1">2 urgent</p>
            </CardContent>
          </Card>
        </div>

        <div className="mt-8">
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start gap-4 pb-4 border-b">
                  <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-semibold text-primary">JD</span>
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">John Doe - Annual Checkup</p>
                    <p className="text-sm text-muted-foreground">Completed today at 10:30 AM</p>
                  </div>
                </div>

                <div className="flex items-start gap-4 pb-4 border-b">
                  <div className="w-10 h-10 bg-accent/10 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-semibold text-accent">SM</span>
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">Sarah Miller - Lab Results</p>
                    <p className="text-sm text-muted-foreground">Reviewed today at 9:15 AM</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-semibold text-primary">RJ</span>
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">Robert Johnson - Prescription Refill</p>
                    <p className="text-sm text-muted-foreground">Approved yesterday at 4:45 PM</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
