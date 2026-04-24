package op

import (
	"context"
	"testing"

	"github.com/bestruirui/octopus/internal/model"
)

func TestGroupGetEnabledMapReturnsFirstRegexMatchByID(t *testing.T) {
	groupCache.Clear()
	channelCache.Clear()
	t.Cleanup(func() {
		groupCache.Clear()
		channelCache.Clear()
	})

	channelCache.Set(1, model.Channel{ID: 1, Enabled: true})
	channelCache.Set(2, model.Channel{ID: 2, Enabled: true})

	groupCache.Set(2, model.Group{
		ID:         2,
		Name:       "broad",
		MatchRegex: "^gpt-.*$",
		Items: []model.GroupItem{
			{ID: 20, GroupID: 2, ChannelID: 2, ModelName: "gpt-4.1"},
		},
	})
	groupCache.Set(1, model.Group{
		ID:         1,
		Name:       "exact",
		MatchRegex: "^gpt-4\\.1$",
		Items: []model.GroupItem{
			{ID: 10, GroupID: 1, ChannelID: 1, ModelName: "gpt-4.1"},
		},
	})

	group, err := GroupGetEnabledMap("gpt-4.1", context.Background())
	if err != nil {
		t.Fatalf("GroupGetEnabledMap returned error: %v", err)
	}
	if group.ID != 1 {
		t.Fatalf("expected first regex match by ID to win, got group %d", group.ID)
	}
	if len(group.Items) != 1 || group.Items[0].ChannelID != 1 {
		t.Fatalf("expected items from group 1, got %+v", group.Items)
	}
}

func TestGroupGetEnabledMapDoesNotFallbackToPrefix(t *testing.T) {
	groupCache.Clear()
	channelCache.Clear()
	t.Cleanup(func() {
		groupCache.Clear()
		channelCache.Clear()
	})

	groupCache.Set(1, model.Group{
		ID:   1,
		Name: "gpt",
		Items: []model.GroupItem{
			{ID: 10, GroupID: 1, ChannelID: 1, ModelName: "gpt-4.1"},
		},
	})

	if _, err := GroupGetEnabledMap("gpt-4.1", context.Background()); err == nil {
		t.Fatal("expected no group match without match_regex")
	}
}

func TestGroupGetEnabledMapFallsBackToExactGroupNameWhenRegexEmpty(t *testing.T) {
	groupCache.Clear()
	channelCache.Clear()
	t.Cleanup(func() {
		groupCache.Clear()
		channelCache.Clear()
	})

	channelCache.Set(1, model.Channel{ID: 1, Enabled: true})

	groupCache.Set(1, model.Group{
		ID:   1,
		Name: "gpt-4.1",
		Items: []model.GroupItem{
			{ID: 10, GroupID: 1, ChannelID: 1, ModelName: "gpt-4.1"},
		},
	})

	group, err := GroupGetEnabledMap("gpt-4.1", context.Background())
	if err != nil {
		t.Fatalf("GroupGetEnabledMap returned error: %v", err)
	}
	if group.ID != 1 {
		t.Fatalf("expected exact group-name match, got group %d", group.ID)
	}
}
