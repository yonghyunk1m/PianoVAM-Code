import test from 'node:test';
import assert from 'node:assert/strict';

import {
  auditCategoryForNote,
  explicitAuditPriority,
  hardReasonsForNote,
  isAuditNote,
  isExplicitAuditNote,
} from '../src/auditCategories.js';


const explicitOnly = {
  is_hard: false,
  hard_reasons: [],
};


test('explicit-only physical notes are audit notes', () => {
  const note = {
    ...explicitOnly,
    physical_must_alert: true,
    physical_reasons: ['same_finger_simultaneous_keys'],
  };

  assert.equal(isExplicitAuditNote(note), true);
  assert.equal(isAuditNote(note), true);
});


test('explicit-only integrity notes are audit notes', () => {
  const note = {
    ...explicitOnly,
    data_integrity_must_resolve: true,
    data_integrity_reasons: ['missing_offset'],
  };

  assert.equal(isExplicitAuditNote(note), true);
  assert.equal(isAuditNote(note), true);
});


test('explicit-only Noinfo context notes are audit notes', () => {
  const note = {
    ...explicitOnly,
    noinfo_context_alert: true,
    noinfo_context_reasons: ['noinfo_context_k3_r2'],
  };

  assert.equal(isExplicitAuditNote(note), true);
  assert.equal(isAuditNote(note), true);
});


test('legacy hard flags and reasons remain audit notes', () => {
  assert.equal(isAuditNote({ is_hard: true, hard_reasons: [] }), true);
  assert.equal(
    isAuditNote({ is_hard: false, hard_reasons: ['fast_jump'] }),
    true,
  );
  assert.deepEqual(
    hardReasonsForNote({ hard_reasons: 'fast_jump,noinfo_cluster' }),
    ['fast_jump', 'noinfo_cluster'],
  );
});


test('category extraction provides labels and precomputed reasons', () => {
  assert.deepEqual(
    auditCategoryForNote({
      physical_must_alert: true,
      physical_reasons: ['same_finger_simultaneous_keys'],
    }),
    {
      id: 'physical',
      label: 'Physical must-alert',
      reasons: ['same_finger_simultaneous_keys'],
      reason: 'physical must-alert: same_finger_simultaneous_keys',
      priority: 110,
    },
  );
  assert.deepEqual(
    auditCategoryForNote({
      data_integrity_must_resolve: true,
      data_integrity_reasons: ['missing_offset'],
    }),
    {
      id: 'integrity',
      label: 'Data integrity',
      reasons: ['missing_offset'],
      reason: 'data integrity: missing_offset',
      priority: 105,
    },
  );
  assert.deepEqual(
    auditCategoryForNote({
      noinfo_context_alert: true,
      noinfo_context_reasons: ['noinfo_context_k3_r2'],
    }),
    {
      id: 'noinfo_context',
      label: 'Near Noinfo region',
      reasons: ['noinfo_context_k3_r2'],
      reason: 'near Noinfo region: noinfo_context_k3_r2',
      priority: 98,
    },
  );
});


test('explicit priority is physical then integrity then Noinfo', () => {
  const physical = explicitAuditPriority({
    physical_must_alert: true,
    physical_reasons: [],
  });
  const integrity = explicitAuditPriority({
    data_integrity_must_resolve: true,
    data_integrity_reasons: [],
  });
  const noinfo = explicitAuditPriority({
    noinfo_context_alert: true,
    noinfo_context_reasons: [],
  });

  assert.ok(physical.score > integrity.score);
  assert.ok(integrity.score > noinfo.score);
  assert.deepEqual(
    [physical.score, integrity.score, noinfo.score],
    [110, 105, 98],
  );
});


test('physical category wins when multiple explicit flags are present', () => {
  const category = auditCategoryForNote({
    physical_must_alert: true,
    physical_reasons: ['physical'],
    data_integrity_must_resolve: true,
    data_integrity_reasons: ['integrity'],
    noinfo_context_alert: true,
    noinfo_context_reasons: ['noinfo'],
  });

  assert.equal(category.id, 'physical');
  assert.equal(explicitAuditPriority({
    physical_must_alert: true,
    data_integrity_must_resolve: true,
  }).score, 110);
});
