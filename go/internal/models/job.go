package models

import (
	"time"

	"github.com/google/uuid"
)

type JobStatus string

const (
	StatusPending    JobStatus = "pending"
	StatusScheduled  JobStatus = "scheduled"
	StatusProcessing JobStatus = "processing"
	StatusDone       JobStatus = "done"
	StatusFailed     JobStatus = "failed"
)

// Unit of worker that is passed through the system
type InferenceJob struct {
	ID      string `json:"id"`
	Prompt  string `json:"prompt"`
	ModelID string `json:"model_id"`

	Status      JobStatus `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	ScheduledAt time.Time `json:"scheduled_at"`
	CompletedAt time.Time `json:"completed_at"`

	Priority       int       `json:"priority"`         
    Deadline       time.Time `json:"deadline"`         
    AssignedWorker string    `json:"assigned_worker"`

	Output    string `json:"output"`
	ErrorMsg  string `json:"error,omitempty"`
	LatencyMs int64  `json:"latency_ms"`
}

// GPU inference worker node
type Worker struct {
	ID string `json:"id"`
	Endpoint string `json:"endpoint"`

	Healthy string `json:"healthy"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
}

type SubmitRequest struct {
    Prompt       string `json:"prompt"`
    SystemPrompt string `json:"system_prompt"`
    ModelID      string `json:"model_id"`
    Priority     int    `json:"priority"`
    DeadlineMs   int    `json:"deadline_ms"`
}

type SubmitResponse struct {
    JobID     string `json:"job_id"`
    Status    string `json:"status"`
    PollURL   string `json:"poll_url"`
}

type StatusResponse struct {
    JobID       string    `json:"job_id"`
    Status      JobStatus `json:"status"`
    Output      string    `json:"output,omitempty"`
    LatencyMs   int64     `json:"latency_ms,omitempty"`
    AssignedTo  string    `json:"assigned_to,omitempty"`
    Error       string    `json:"error,omitempty"`
}
