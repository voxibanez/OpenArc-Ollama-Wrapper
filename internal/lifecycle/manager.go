package lifecycle

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/compat"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/config"
	"github.com/voxibanez/OpenArc-Ollama-Wrapper/internal/openarc"
)

type OpenArc interface {
	Load(context.Context, openarc.ModelLoadConfig) error
	Unload(context.Context, string) error
	Status(context.Context) (openarc.Status, error)
	StartDownload(context.Context, string) error
	Downloads(context.Context) ([]openarc.DownloadTask, error)
}

type Options struct {
	MaxLoadedModels      int
	DefaultKeepAlive     time.Duration
	CheckInterval        time.Duration
	DownloadPollInterval time.Duration
}

type Manager struct {
	manifest *config.Manifest
	openarc  OpenArc
	opts     Options
	mu       sync.Mutex
	states   map[string]*state
	stop     chan struct{}
}

type state struct {
	model     *config.Model
	mu        sync.Mutex
	loaded    bool
	active    int
	lastUsed  time.Time
	expiresAt time.Time
}

func NewManager(manifest *config.Manifest, openArc OpenArc, opts Options) *Manager {
	if opts.MaxLoadedModels < 1 {
		opts.MaxLoadedModels = 1
	}
	if opts.DefaultKeepAlive == 0 {
		opts.DefaultKeepAlive = time.Minute
	}
	if opts.CheckInterval == 0 {
		opts.CheckInterval = 10 * time.Second
	}
	if opts.DownloadPollInterval == 0 {
		opts.DownloadPollInterval = time.Second
	}
	m := &Manager{
		manifest: manifest,
		openarc:  openArc,
		opts:     opts,
		states:   map[string]*state{},
		stop:     make(chan struct{}),
	}
	go m.reaper()
	return m
}

func (m *Manager) EnsureLoaded(ctx context.Context, model *config.Model, keepAlive *time.Duration) (*Lease, error) {
	st := m.getState(model)
	st.mu.Lock()
	defer st.mu.Unlock()

	st.active++
	lease := &Lease{manager: m, state: st, model: model, keepAlive: m.keepAlive(keepAlive)}

	if st.loaded {
		st.lastUsed = time.Now()
		return lease, nil
	}
	if !compat.LocalPathLooksPresent(model.ModelPath) {
		if err := m.downloadAndWait(ctx, model); err != nil {
			st.active--
			return nil, err
		}
	}
	if err := compat.CheckLocalPath(model.ModelPath); err != nil {
		st.active--
		return nil, fmt.Errorf("local OpenVINO validation failed for %s: %w", model.Name, err)
	}
	if loaded, err := m.openArcHasModelLoaded(ctx, model.Name); err != nil {
		st.active--
		return nil, err
	} else if loaded {
		st.loaded = true
		st.lastUsed = time.Now()
		return lease, nil
	}
	if err := m.evictIfNeeded(ctx, model.Name); err != nil {
		st.active--
		return nil, err
	}
	if err := m.openarc.Load(ctx, openarc.ModelLoadConfig{
		ModelPath:     model.ModelPath,
		ModelName:     model.Name,
		ModelType:     model.ModelType,
		Engine:        model.Engine,
		Device:        model.Device,
		RuntimeConfig: model.RuntimeConfig,
	}); err != nil {
		st.active--
		return nil, err
	}
	st.loaded = true
	st.lastUsed = time.Now()
	return lease, nil
}

func (m *Manager) UnloadNow(ctx context.Context, model *config.Model) error {
	st := m.getState(model)
	st.mu.Lock()
	defer st.mu.Unlock()
	if !st.loaded {
		return nil
	}
	if err := m.openarc.Unload(ctx, model.Name); err != nil {
		return err
	}
	st.loaded = false
	st.expiresAt = time.Time{}
	return nil
}

func (m *Manager) MarkDownloadedAndLoadable(ctx context.Context, model *config.Model) error {
	if err := compat.CheckLocalPath(model.ModelPath); err != nil {
		return err
	}
	return nil
}

func (m *Manager) getState(model *config.Model) *state {
	m.mu.Lock()
	defer m.mu.Unlock()
	if st, ok := m.states[model.Name]; ok {
		return st
	}
	st := &state{model: model}
	m.states[model.Name] = st
	return st
}

func (m *Manager) keepAlive(value *time.Duration) time.Duration {
	if value != nil {
		return *value
	}
	return m.opts.DefaultKeepAlive
}

func (m *Manager) downloadAndWait(ctx context.Context, model *config.Model) error {
	if err := m.openarc.StartDownload(ctx, model.HFRepo); err != nil {
		return err
	}
	ticker := time.NewTicker(m.opts.DownloadPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			tasks, err := m.openarc.Downloads(ctx)
			if err != nil {
				return err
			}
			for _, task := range tasks {
				if task.ModelName != model.HFRepo {
					continue
				}
				switch task.Status {
				case "completed":
					return nil
				case "error", "cancelled":
					return fmt.Errorf("download %s for %s", task.Status, model.HFRepo)
				}
			}
		}
	}
}

func (m *Manager) openArcHasModelLoaded(ctx context.Context, modelName string) (bool, error) {
	status, err := m.openarc.Status(ctx)
	if err != nil {
		return false, fmt.Errorf("openarc status check failed before loading %s: %w", modelName, err)
	}
	for _, loaded := range status.Models {
		if loaded.ModelName == modelName && loaded.Status != "unloaded" {
			return true, nil
		}
	}
	return false, nil
}

func (m *Manager) evictIfNeeded(ctx context.Context, incoming string) error {
	m.mu.Lock()
	var loaded []*state
	for name, st := range m.states {
		if name == incoming {
			continue
		}
		st.mu.Lock()
		if st.loaded && st.active == 0 {
			loaded = append(loaded, st)
		}
		st.mu.Unlock()
	}
	currentLoaded := len(loaded)
	if st := m.states[incoming]; st != nil && st.loaded {
		currentLoaded++
	}
	limit := m.opts.MaxLoadedModels
	m.mu.Unlock()
	for currentLoaded >= limit && len(loaded) > 0 {
		victimIdx := 0
		for i := 1; i < len(loaded); i++ {
			if loaded[i].lastUsed.Before(loaded[victimIdx].lastUsed) {
				victimIdx = i
			}
		}
		victim := loaded[victimIdx]
		if err := m.UnloadNow(ctx, victim.model); err != nil {
			return err
		}
		loaded = append(loaded[:victimIdx], loaded[victimIdx+1:]...)
		currentLoaded--
	}
	return nil
}

func (m *Manager) reaper() {
	ticker := time.NewTicker(m.opts.CheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-ticker.C:
			m.unloadExpired()
		}
	}
}

func (m *Manager) unloadExpired() {
	now := time.Now()
	m.mu.Lock()
	states := make([]*state, 0, len(m.states))
	for _, st := range m.states {
		states = append(states, st)
	}
	m.mu.Unlock()
	for _, st := range states {
		st.mu.Lock()
		expired := st.loaded && st.active == 0 && !st.expiresAt.IsZero() && now.After(st.expiresAt)
		st.mu.Unlock()
		if expired {
			_ = m.UnloadNow(context.Background(), st.model)
		}
	}
}

type Lease struct {
	manager   *Manager
	state     *state
	model     *config.Model
	keepAlive time.Duration
	once      sync.Once
}

func (l *Lease) Done(ctx context.Context) error {
	var err error
	l.once.Do(func() {
		l.state.mu.Lock()
		l.state.active--
		l.state.lastUsed = time.Now()
		if l.keepAlive == 0 {
			l.state.mu.Unlock()
			err = l.manager.UnloadNow(ctx, l.model)
			return
		}
		l.state.expiresAt = l.state.lastUsed.Add(l.keepAlive)
		l.state.mu.Unlock()
	})
	return err
}
