"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"
import { Header } from "@/components/header"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export default function UserDashboard() {
  const { isAuthenticated, userRole } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isAuthenticated || userRole !== "user") {
      router.push("/login")
    }
  }, [isAuthenticated, userRole, router])

  if (!isAuthenticated || userRole !== "user") {
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Patient Portal</h1>
          <p className="text-muted-foreground">Welcome back, Jane Anderson</p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Upcoming Appointments</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">2</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Prescriptions</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">3</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Lab Results</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">1 New</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Messages</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">2</p>
            </CardContent>
          </Card>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Upcoming Appointments</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start justify-between pb-4 border-b">
                  <div>
                    <p className="font-medium">Dr. Michael Chen</p>
                    <p className="text-sm text-muted-foreground">Annual Physical Exam</p>
                    <p className="text-sm text-muted-foreground mt-1">March 15, 2025 at 2:00 PM</p>
                  </div>
                  <Button variant="outline" size="sm">
                    Reschedule
                  </Button>
                </div>

                <div className="flex items-start justify-between">
                  <div>
                    <p className="font-medium">Dr. Sarah Williams</p>
                    <p className="text-sm text-muted-foreground">Follow-up Consultation</p>
                    <p className="text-sm text-muted-foreground mt-1">March 22, 2025 at 10:30 AM</p>
                  </div>
                  <Button variant="outline" size="sm">
                    Reschedule
                  </Button>
                </div>
              </div>

              <Button className="w-full mt-6">Book New Appointment</Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Lab Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start justify-between pb-4 border-b">
                  <div>
                    <p className="font-medium">Complete Blood Count (CBC)</p>
                    <p className="text-sm text-muted-foreground">Ordered by Dr. Michael Chen</p>
                    <p className="text-sm text-muted-foreground mt-1">March 1, 2025</p>
                  </div>
                  <Button variant="outline" size="sm">
                    View
                  </Button>
                </div>

                <div className="flex items-start justify-between pb-4 border-b">
                  <div>
                    <p className="font-medium">Lipid Panel</p>
                    <p className="text-sm text-muted-foreground">Ordered by Dr. Michael Chen</p>
                    <p className="text-sm text-muted-foreground mt-1">February 15, 2025</p>
                  </div>
                  <Button variant="outline" size="sm">
                    View
                  </Button>
                </div>

                <div className="flex items-start justify-between">
                  <div>
                    <p className="font-medium">Thyroid Function Test</p>
                    <p className="text-sm text-muted-foreground">Ordered by Dr. Sarah Williams</p>
                    <p className="text-sm text-muted-foreground mt-1">January 28, 2025</p>
                  </div>
                  <Button variant="outline" size="sm">
                    View
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Active Prescriptions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="pb-4 border-b">
                  <div className="flex items-start justify-between mb-2">
                    <p className="font-medium">Lisinopril 10mg</p>
                    <Button variant="outline" size="sm">
                      Refill
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground">Take once daily</p>
                  <p className="text-sm text-muted-foreground">Refills remaining: 2</p>
                </div>

                <div className="pb-4 border-b">
                  <div className="flex items-start justify-between mb-2">
                    <p className="font-medium">Metformin 500mg</p>
                    <Button variant="outline" size="sm">
                      Refill
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground">Take twice daily with meals</p>
                  <p className="text-sm text-muted-foreground">Refills remaining: 1</p>
                </div>

                <div>
                  <div className="flex items-start justify-between mb-2">
                    <p className="font-medium">Atorvastatin 20mg</p>
                    <Button variant="outline" size="sm">
                      Refill
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground">Take once daily at bedtime</p>
                  <p className="text-sm text-muted-foreground">Refills remaining: 3</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Messages</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="pb-4 border-b">
                  <div className="flex items-start justify-between mb-2">
                    <p className="font-medium">Dr. Michael Chen</p>
                    <span className="text-xs bg-accent text-accent-foreground px-2 py-1 rounded">New</span>
                  </div>
                  <p className="text-sm text-muted-foreground">Your lab results are ready for review...</p>
                  <p className="text-xs text-muted-foreground mt-1">2 hours ago</p>
                </div>

                <div className="pb-4 border-b">
                  <div className="flex items-start justify-between mb-2">
                    <p className="font-medium">Appointment Reminder</p>
                    <span className="text-xs bg-accent text-accent-foreground px-2 py-1 rounded">New</span>
                  </div>
                  <p className="text-sm text-muted-foreground">You have an appointment tomorrow at 2:00 PM...</p>
                  <p className="text-xs text-muted-foreground mt-1">1 day ago</p>
                </div>

                <div>
                  <p className="font-medium mb-2">Prescription Refill Approved</p>
                  <p className="text-sm text-muted-foreground">Your prescription for Lisinopril has been approved...</p>
                  <p className="text-xs text-muted-foreground mt-1">3 days ago</p>
                </div>
              </div>

              <Button className="w-full mt-6">View All Messages</Button>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
